    %% Step 01 -- Read Geotiff and Meteorological Variables
    % Create satellite-derived, time-series data of chl-a
    % Input required:(a) Directory that contains Geotif data and (b) PointOfInterest = [latitude, longitude]
    Observations = ReadGeoTiff(DirectoryName,PointOfInterest);
    % Create time-series data of meteorological variables
    % Input required:(a) utc offset for the point of interest and (b) PointOfInterest = [latitude, longitude]
    AirTemperature = ReadBiasAdjustedAirTemp(utcoffset,PointOfInterest);
    SolarRadiation = ReadBiasAdjustedRad(utcoffset,PointOfInterest);
    TotalPrecipitation = ReadBiasAdjustedTotPr(utcoffset,PointOfInterest);
    WindSpeed = ReadBiasAdjustedWind(utcoffset,PointOfInterest);
    MeteoInp = [AirTemperature(:,1) AirTemperature(:,2) SolarRadiation(:,2) TotalPrecipitation(:,2) WindSpeed(:,2)];
    MeteoInp.Properties.VariableNames(5) = "windspeed";
    MeteoInp.Properties.VariableNames(4) = "tp";
    MeteoInp.Properties.VariableNames(2) = "t2m";
    MeteoInp.Properties.VariableNames(3) = "ssrd";
    MeteoInp.Properties.VariableNames(1) = "time";
    % Create time-series data of hydrological variables
    % read Total nitrogen
    cctn = ReadHydroData(HydroFileName,NoCatchments); % the correct file path and name should be provided 
    % read Total phosphorus
    cctp = ReadHydroData(HydroFileName,NoCatchments); % the correct file path and name should be provided 
    % total inflow
    cout = ReadHydroData(HydroFileName,NoCatchments); % the correct file path and name should be provided 
    % For more than one catchments more than one variables must be concantenated for each catchment to form HydroInp
    HydroInp = [cctn(:,1) cctn(:,2) cctp(:,2) cout(:,2)]; 
    HydroInp.Properties.VariableNames(4) = "cout";
    HydroInp.Properties.VariableNames(2) = "cctn";
    HydroInp.Properties.VariableNames(3) = "cctp";
    HydroInp.Properties.VariableNames(1) = "time";
    %% Step 02 -- PreProcess Chl Data
    Observations = FillnSmooth(Observations);
    %% Step 03 -- Find optimal width of the sliding windows
    [window] = OptimalWindow(CaseStudy,HydroInp,MeteoInp,Observations,point,forecast_horizon);
    % create final training dataset
    [predictors,target] = CreateTrainingData(CaseStudy,HydroInp,MeteoInp,Observations,point,window,forecast_horizon);
    [norm_data,C,S] = normalize([predictors target]);
    %% Step 04 -- Feature Selection
	PredictorList = FeatureSelection(norm_data(:,1:end-1),norm_data(:,end));
    %% Step 05 -- Model training out-of-bag errors
    [qrf01,bestHyperparametersRF,~] = TrainRF(norm_data(:,1:end-1),norm_data(:,end),PredictorList);
    %% Step 06 - Model training using out-of-bag quantile errors
    [qrf02,bestHyperparametersQRF,~] = TrainQRF(norm_data(:,1:end-1),norm_data(:,end),PredictorList);
   %% Step 07 - Model comparison
    [RS1,AW1,RS2,AW2] = EvalUnc(qrf01,qrf02,norm_data(:,1:end-1),norm_data(:,end),0.90);
    [~,mean_crpsQRF01] = Crps(qrf01,norm_data(:,1:end-1),norm_data(:,end));
    [~,mean_crpsQRF02] = Crps(qrf02,norm_data(:,1:end-1),norm_data(:,end));
    %%

function Observations = ReadGeoTiff(DirectoryName,PointOfInterest)
%% detect all geotifs residing in the directory
cd(DirectoryName)
listChl=dir('*.tif*');
for i=1:length(listChl)
    [~,filename,~] = fileparts(listChl(i).name);
    p = strfind(filename,'20');
    tc(i,1) = datetime(filename(p:p+7),'InputFormat','yyyyMMdd'); % find dates of every tif
end
%% Estimate Chl-a values and weights
[t_chl,ia,~]=unique(tc);
chlvalue = zeros(length(t_chl),1);
chlrstd = zeros(length(t_chl),1);
clear tc

for i=1:length(ia)
    ind = ia(i);
    [~,filename,extension] = fileparts(listChl(ind).name);
    [A,R] = readgeoraster(strcat(filename,extension)); % read the tif
     A = double(A);
     A(A<0) = NaN;
     xx = (R.XWorldLimits(1)+0.5*R.CellExtentInWorldX):R.CellExtentInWorldX:(R.XWorldLimits(2)-0.5*R.CellExtentInWorldX);
     yy = (R.YWorldLimits(1)+0.5*R.CellExtentInWorldY):R.CellExtentInWorldY:(R.YWorldLimits(2)-0.5*R.CellExtentInWorldY);
     [xval,yval] = meshgrid(xx,yy); 
     yval = flip(yval,1);
     Ypoints = [reshape(xval,numel(xval),1), reshape(yval,numel(yval),1)];
     Avalues = reshape(A,numel(A),1);
     [Idx,~] = rangesearch(PointOfInterest,Ypoints,200); % checking within a 200m radius
     identity = ~cellfun(@isempty,Idx);
     Avalues = Avalues(identity);
     chlvalue(i,1) = mean(Avalues(~isnan(Avalues))); % averaging nearby points
     chlrstd(i,1)= std(Avalues,'omitnan')./chlvalue(i,1);
    clear Idx D xx yy xval yval A R identity Avalues Ypoints
end

t_chl = t_chl(~isnan(chlvalue));
chlrstd = chlrstd(~isnan(chlvalue));
chlvalue = chlvalue(~isnan(chlvalue));
Observations = table(t_chl,chlvalue,1./chlrstd);
Observations.Properties.VariableNames(2) = "value";
Observations.Properties.VariableNames(1) = "time";
Observations.Properties.VariableNames(3) = "weight";
end

function AirTinput = ReadBiasAdjustedAirTemp(utcoffset,point)
    %% Read all available netcdf files in the folder
    list=dir('*nc*');
    %% read files in a loop
     AirTinput = table();
    for i=1:length(list)
       % read nc files
        [~,filename,extension] = fileparts(list(i).name);
        ncfile = strcat(filename,extension);
        % read dates
        timeData  = ncread(ncfile,'time');
        DatesECMWF = datetime(1900,1,1) + hours(timeData+utcoffset) ; %
        DatesECMWF.Format = 'yyyy-MM-dd HH:mm';
        % read longitude and latitude
        latitude = ncread(ncfile,'lat');
        longitude = ncread(ncfile,'lon');
        [x,y] = meshgrid(latitude,longitude);
        % read air temperature data and find values closest to the center of AOI
        airt  = ncread(ncfile,'Tair');  
        airtdata = zeros(size(airt,3),1);
        for j=1:length(airtdata)
            airtdata(j,1)= interp2(x,y,squeeze(squeeze(airt(:,:,j))),point(1),point(2),'nearest')-273.15;
        end
        % add data to the existing table
        AirTinput = [AirTinput;table(DatesECMWF,airtdata)];
        clear airt airtdata DatesECMWF
    end
end

function SwdownInput = ReadBiasAdjustedRad(utcoffset,point)

%% Read all available netcdf files in the folder
list=dir('*nc*');
%% read files in a loop
SwdownInput = table();
for i=1:length(list)
    % read nc files
    [~,filename,extension] = fileparts(list(i).name);
    ncfile = strcat(filename,extension);
    % read dates
    timeData  = ncread(ncfile,'time');
    DatesECMWF = datetime(1900,1,1) + hours(timeData+utcoffset) ; % 
    DatesECMWF.Format = 'yyyy-MM-dd HH:mm';
    % read loongitude and latitude
    latitude = ncread(ncfile,'lat');
    longitude = ncread(ncfile,'lon');
    [x,y] = meshgrid(latitude,longitude);
    % read radiation data and find values closest to the center of AOI
    swdown  = ncread(ncfile,'SWdown');  
    swdata = zeros(size(swdown,3),1);
    for j=1:length(swdata)
        swdata(j,1)= interp2(x,y,squeeze(squeeze(swdown(:,:,j))),point(1),point(2),'nearest');
    end
    % add data to the existing table
    SwdownInput = [SwdownInput;table(DatesECMWF,swdata)];
    clear swdata swdown DatesECMWF
end
end

function TotP = ReadBiasAdjustedTotPr(utcoffset,point)
    %% Read all available netcdf files in the folder
    list=dir('*nc*');
    %% read files in a loop
    TotP = table();
    for i=1:length(list)
        % read nc files
        [~,filename,extension] = fileparts(list(i).name);
        ncfile = strcat(filename,extension);
         % read dates
        timeData  = ncread(ncfile,'time');
        DatesECMWF = datetime(1900,1,1) + hours(timeData+utcoffset) ; % utc offset =-4
        DatesECMWF.Format = 'yyyy-MM-dd HH:mm';
        % read longitude and latitude
        latitude = ncread(ncfile,'lat');
        longitude = ncread(ncfile,'lon');
        [x,y] = meshgrid(latitude,longitude);
        % read rainfall data and find values closest to the center of AOI
        rain  = ncread(ncfile,'Rainf');
        snow  = ncread(ncfile,'Snowf');
        raindata = zeros(size(rain,3),1);
        snowdata = zeros(size(snow,3),1);
        for j=1:length(raindata)
            raindata(j,1)= interp2(x,y,squeeze(squeeze(rain(:,:,j))),point(1),point(2),'nearest').*3.6; % [kg m-2 s] to [m]
        end
        for j=1:length(snowdata)
            snowdata(j,1)= interp2(x,y,squeeze(squeeze(snow(:,:,j))),point(1),point(2),'nearest').*3.6; % [kg m-2 s] to [m]
        end
        tpdata = raindata + snowdata;
        % add data to the existing table
        TotP = [TotP;table(DatesECMWF,tpdata)];
        clear raindata snowdata tpdata snow rain DatesECMWF

    end
end

function WindInput = ReadBiasAdjustedWind(utcoffset,point)
    %% Read all available netcdf files in the folder
    list=dir('*nc*');
    %% read files in a loop
    WindInput = table();
    for i=1:length(list)
        % read the nc file
        [~,filename,extension] = fileparts(list(i).name);
        ncfile = strcat(filename,extension);
        % read dates
        timeData  = ncread(ncfile,'time');
        DatesECMWF = datetime(1900,1,1) + hours(timeData+utcoffset) ; %
        DatesECMWF.Format = 'yyyy-MM-dd HH:mm';
        % read longitude and latitude
        latitude = ncread(ncfile,'lat');
        longitude = ncread(ncfile,'lon');
        [x,y] = meshgrid(latitude,longitude);
        % read wind data and find values closest to the center of AOI
        wind  = ncread(ncfile,'Wind');  
        winddata = zeros(size(wind,3),1);
        for j=1:length(winddata)
            winddata(j,1)= interp2(x,y,squeeze(squeeze(wind(:,:,j))),point(1),point(2),'nearest');
        end
        % add data to the existing table
        WindInput = [WindInput;table(DatesECMWF,winddata)];
        clear winddata wind DatesECMWF
    end
end

function HydroVariable = ReadHydroData(file,NoCatchments)
    %% Import data from text file
    %% Set up the Import Options and import the data
    opts = delimitedTextImportOptions("NumVariables", NoCatchments);
    % Specify range and delimiter
    opts.DataLines = [3, Inf];
    opts.Delimiter = "\t";
    % Specify column names and types
    opts.VariableTypes(1) = "datetime";
    for i=1:legnth(opts.VariableTypes)
        opts.VariableTypes(i) = "double";
    end
    % Specify file level properties
    opts.ExtraColumnsRule = "ignore";
    opts.EmptyLineRule = "read";
    % Specify variable properties
    opts = setvaropts(opts, "Date", "InputFormat", "yyyy-MM-dd");
    % Import the data
    HydroVariable = readtable(file, opts);
end

function Observations = FillnSmooth(Observations)
	% Fill missing data
	Observations = fillmissing(Observations,"makima","MaxGap",days(3),...
	    "DataVariables","value");
	% Smooth input data
	Observations = smoothdata(Observations,"sgolay",days(7),"Degree",2,...
	    "DataVariables","value");
end

function [window] = OptimalWindow(CaseStudy,HydroInp,MeteoInp,Observations,forecast_horizon)
        %% Estimate optimal window using a grid-based search
        [Window1,Window2] = meshgrid(7:1:14,7:1:14);
        loss = zeros(length(Window1),length(Window2));
        for i =1:length(Window1)
            for j = 1:length(Window2)
                wnd = [Window1(i) Window2(j)];
                [predictors,target] = CreateTrainingData(CaseStudy,HydroInp,MeteoInp,Observations,1,wnd,forecast_horizon);
                [predictors,~,~] = normalize(predictors);
                minMinLS = max(5,round(size(predictors(~isnan(predictors(:,end-1))),1)/40));
                maxMinLS = round(size(predictors(~isnan(predictors(:,end-1))),1)/10);
                maxnumPTS = size(predictors,2)-1; 
                maxnumsplits = height(Observations)-1;
                minLS = optimizableVariable('minLS',[minMinLS,maxMinLS],'Type','integer');
                numPTS = optimizableVariable('numPTS',[2,maxnumPTS],'Type','integer');
                NumSplits = optimizableVariable('NumSplits',[1,maxnumsplits],'Type','integer');
                numTrees = optimizableVariable('numTrees',[20,500],'Type','integer');
                hyperparametersRF = [minLS;numPTS;NumSplits;numTrees];
                %%
                results = bayesopt(@(params)ObjFun(params,predictors,target),...
                            hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',30,...
                            'Verbose',0,'UseParallel',true,'PlotFcn',[]);
                loss(i,j) = results.MinObjective;
    
                clear Window predictors target weights results
            end
        end 
        minimum = min(min(loss));
        [w1,w2]=find(loss==minimum);
        window = [Window1(w1,w2),Window2(w1,w2)];
end

function [predictors,target] = CreateTrainingData(CaseStudy,HydroInp,MeteoInp,Observations,point,window,forecast_horizon)
if contains(CaseStudy,'us-harsha') == 1
    %%
    tn = zeros(length(Observations.time),1);
    tp = zeros(length(Observations.time),1);
    for i=1:length(Observations.time)
        tf = isbetween(HydroInp.time,Observations.time(i)-window(1)-forecast_horizon,Observations.time(i)-forecast_horizon);
        mov_cout1 = HydroInp.cout01(tf);
        mov_cout2 = HydroInp.cout02(tf);       
        %
        mov_cctn1 = HydroInp.cctn01(tf);
        mov_cctn2 = HydroInp.cctn02(tf);
        mov_cctp1 = HydroInp.cctp01(tf);
        mov_cctp2 = HydroInp.cctp02(tf);
        %
        tn(i,1) = sum((mov_cctn1.*mov_cout1)*86400*1e-6)+sum((mov_cctn2.*mov_cout2)*86400*1e-6);  % tot kg
        tp(i,1) = sum((mov_cctp1.*mov_cout1)*86400*1e-6)+sum((mov_cctp2.*mov_cout2)*86400*1e-6);  % tot kg    
    end
    %%
        tot_rad = zeros(length(Observations.time),1);
        airt =zeros(length(Observations.time),1);
        tprec = zeros(length(Observations.time),1);
        Wind = zeros(length(Observations.time),1);
    for i=1:length(Observations.time)
        tf = isbetween(MeteoInp.time,Observations.time(i)-window(2)-forecast_horizon,Observations.time(i)-forecast_horizon);
        ssrd_temp = MeteoInp.ssrd(tf);
        tot_rad(i,1) = sum(ssrd_temp); % [W/m2]
        clear ssrd_temp daily_step d td
        %
        mov_airt = MeteoInp.t2m(tf);
        airt(i,1) = mean(mov_airt);
        %
        totp_temp = MeteoInp.tp(tf);
        tprec(i,1) = sum(totp_temp); % [m]
        clear ssrd_temp daily_step d td        
        ws = MeteoInp.windspeed(tf);
        [f,x] = ecdf(ws);
        if ~isempty(find(x>3,1))
            Wind(i,1) = 1-f(find(x>3,1));
        else
            Wind(i,1) = 0;
        end
    end
        %% Create the dataset
        predictors = table();
        % radiation
        predictors.trad = tot_rad; %1
        % air temp
        predictors.airt = airt;%2
        % total precipitation	
        predictors.tprec = tprec;    %3
        % wind
        predictors.Wind = Wind; %4      
        % nutrients
        predictors.tn = (tn); %6
        predictors.tp = (tp);%7
        %
        target = Observations.value;
        predictors = table2array(predictors);
elseif contains(CaseStudy,'au-hume') == 1
    
    if point ==1 || point ==2
        tn01 = zeros(length(Observations.time),1);
        tp01 = zeros(length(Observations.time),1);
        tn02 = zeros(length(Observations.time),1);
        tp02 = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(HydroInp.time,Observations.time(i)-window(1)-forecast_horizon,Observations.time(i)-forecast_horizon);
            mov_cout1 = HydroInp.cout02(tf);
            mov_cout2 = HydroInp.cout04(tf);       
            %
            mov_cctn1 = HydroInp.cctn02(tf);
            mov_cctn2 = HydroInp.cctn04(tf);
            mov_cctp1 = HydroInp.cctp02(tf);
            mov_cctp2 = HydroInp.cctp04(tf);
            %
            tn01(i,1) = sum((mov_cctn1.*mov_cout1)*86400*1e-6);  % tot kg
            tp01(i,1) = sum((mov_cctp1.*mov_cout1)*86400*1e-6);  % tot kg
            tn02(i,1) = sum((mov_cctn2.*mov_cout2)*86400*1e-6);  % tot kg
            tp02(i,1) = sum((mov_cctp2.*mov_cout2)*86400*1e-6);  % tot kg
        end 
        %%
        tot_rad = zeros(length(Observations.time),1);
        airt =zeros(length(Observations.time),1);
        tprec = zeros(length(Observations.time),1);
        Wind = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(MeteoInp.time,Observations.time(i)-window(2)-forecast_horizon,Observations.time(i)-forecast_horizon);
            ssrd_temp = MeteoInp.ssrd(tf);
            tot_rad(i,1) = sum(ssrd_temp); % [W/m2]
            clear ssrd_temp daily_step d td
            %
            mov_airt = MeteoInp.t2m(tf);
            airt(i,1) = mean(mov_airt);
            %
            totp_temp = MeteoInp.tp(tf);
            tprec(i,1) = sum(totp_temp); % [m]
            clear ssrd_temp daily_step d td        
            %
            ws = MeteoInp.windspeed(tf);
            [f,x] = ecdf(ws);
            if ~isempty(find(x>3,1))
                Wind(i,1) = 1-f(find(x>3,1));
            else
                Wind(i,1) = 0;
            end
        end
        %% Create the dataset
        predictors = table();
        % radiation
        predictors.trad = tot_rad; %1
        % air temp
        predictors.airt = airt;%2
        % total precipitation	
        predictors.tprec = tprec;    %3
        % wind
        predictors.Wind = Wind; %4 
        % nutrients
        predictors.tn01 = (tn01); 
        predictors.tn02 = (tn02); 
        predictors.tp01 = (tp01);
        predictors.tp02 = (tp02);
        %
        target = Observations.value;
        predictors = table2array(predictors);
    elseif point ==3
        tn01 = zeros(length(Observations.time),1);
        tp01 = zeros(length(Observations.time),1);
        tn02 = zeros(length(Observations.time),1);
        tp02 = zeros(length(Observations.time),1);
        tn03 = zeros(length(Observations.time),1);
        tp03 = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(HydroInp.time,Observations.time(i)-window(1)-forecast_horizon,Observations.time(i)-forecast_horizon);
            mov_cout1 = HydroInp.cout01(tf);
            mov_cout2 = HydroInp.cout02(tf);
            mov_cout3 = HydroInp.cout04(tf);
            %
            mov_cctn1 = HydroInp.cctn01(tf);
            mov_cctn2 = HydroInp.cctn02(tf);
            mov_cctn3 = HydroInp.cctn04(tf);
            mov_cctp1 = HydroInp.cctp01(tf);
            mov_cctp2 = HydroInp.cctp02(tf);
            mov_cctp3 = HydroInp.cctp04(tf);
            %
            tn01(i,1) = sum((mov_cctn1.*mov_cout1)*86400*1e-6);  % tot kg
            tp01(i,1) = sum((mov_cctp1.*mov_cout1)*86400*1e-6);  % tot kg
            tn02(i,1) = sum((mov_cctn2.*mov_cout2)*86400*1e-6);  % tot kg
            tp02(i,1) = sum((mov_cctp2.*mov_cout2)*86400*1e-6);  % tot kg
            tn03(i,1) = sum((mov_cctn3.*mov_cout3)*86400*1e-6);  % tot kg
            tp03(i,1) = sum((mov_cctp3.*mov_cout3)*86400*1e-6);  % tot kg
        end 
        %%
        tot_rad = zeros(length(Observations.time),1);
        airt =zeros(length(Observations.time),1);
        tprec = zeros(length(Observations.time),1);
        Wind = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(MeteoInp.time,Observations.time(i)-window(2)-forecast_horizon,Observations.time(i)-forecast_horizon);
            ssrd_temp = MeteoInp.ssrd(tf);
            tot_rad(i,1) = sum(ssrd_temp); % [W/m2]
            clear ssrd_temp daily_step d td
            %
            mov_airt = MeteoInp.t2m(tf);
            airt(i,1) = mean(mov_airt);
            %
            totp_temp = MeteoInp.tp(tf);
            tprec(i,1) = sum(totp_temp); % [m]
            clear ssrd_temp daily_step d td        
            %
            ws = MeteoInp.windspeed(tf);
            [f,x] = ecdf(ws);
            if ~isempty(find(x>3,1))
                Wind(i,1) = 1-f(find(x>3,1));
            else
                Wind(i,1) = 0;
            end
        end
        %% Create the dataset
        predictors = table();
        % radiation
        predictors.trad = tot_rad; %1
        % air temp
        predictors.airt = airt;%2
        % total precipitation	
        predictors.tprec = tprec;    %3
        % wind
        predictors.Wind = Wind; %4 
        % nutrients
        predictors.tn01 = (tn01); 
        predictors.tn02 = (tn02); 
        predictors.tn03 = (tn03);
        predictors.tp01 = (tp01);
        predictors.tp02 = (tp02);
        predictors.tp03 = (tp03);
        %
        target = Observations.value;
        predictors = table2array(predictors);
    else
        tn = zeros(length(Observations.time),1);
        tp = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(HydroInp.time,Observations.time(i)-window(1)-forecast_horizon,Observations.time(i)-forecast_horizon);
            mov_cout1 = HydroInp.cout03(tf);            
            mov_cctn1 = HydroInp.cctn03(tf);
            mov_cctp1 = HydroInp.cctp03(tf);
            %
            tn(i,1) = sum((mov_cctn1.*mov_cout1)*86400*1e-6);  %
            tp(i,1) = sum((mov_cctp1.*mov_cout1)*86400*1e-6);  %
        end 
        %%
        tot_rad = zeros(length(Observations.time),1);
        airt =zeros(length(Observations.time),1);
        tprec = zeros(length(Observations.time),1);
        Wind= zeros(length(Observations.time),1);
    
    for i=1:length(Observations.time)
        tf = isbetween(MeteoInp.time,Observations.time(i)-window(2)-forecast_horizon,Observations.time(i)-forecast_horizon);
        ssrd_temp = MeteoInp.ssrd(tf);
        tot_rad(i,1) = sum(ssrd_temp); % [W/m2]
        clear ssrd_temp daily_step d td
        %
        mov_airt = MeteoInp.t2m(tf);
        airt(i,1) = mean(mov_airt);
        %
        totp_temp = MeteoInp.tp(tf);
        tprec(i,1) = sum(totp_temp); % [m]
        clear ssrd_temp daily_step d td        
        %
            ws = MeteoInp.windspeed(tf);
            [f,x] = ecdf(ws);
            if ~isempty(find(x>3,1))
                Wind(i,1) = 1-f(find(x>3,1));
            else
                Wind(i,1) = 0;
            end
    end
        %% Create the dataset
        predictors = table();
        % radiation
        predictors.trad = tot_rad; %1
        % air temp
        predictors.airt = airt;%2
        % total precipitation	
        predictors.tprec = tprec;    %3
        % wind
        predictors.Wind = Wind; %4 
        % nutriens
        predictors.tn01 = (tn); %6
        predictors.tp01 = (tp);%7
        %
        target = Observations.value;
        predictors = table2array(predictors);
    end
elseif contains(CaseStudy,'it-sardinia') == 1
        tn = zeros(length(Observations.time),1);
        tp = zeros(length(Observations.time),1);
        for i=1:length(Observations.time)
            tf = isbetween(HydroInp.time,Observations.time(i)-window(1)-forecast_horizon,Observations.time(i)-forecast_horizon);
            mov_cout = HydroInp.cout(tf);
            %
            mov_cctn = HydroInp.cctn(tf);
            mov_cctp = HydroInp.cctp(tf);
            %
            tn(i,1) = sum((mov_cctn.*mov_cout)*86400*1e-6);  % tot kg
            tp(i,1) = sum((mov_cctp.*mov_cout)*86400*1e-6);  % tot kg    
        end
    %%
        tot_rad = zeros(length(Observations.time),1);
        airt =zeros(length(Observations.time),1);
        tprec = zeros(length(Observations.time),1);
        Wind = zeros(length(Observations.time),1);
    for i=1:length(Observations.time)
        tf = isbetween(MeteoInp.time,Observations.time(i)-window(2)-forecast_horizon,Observations.time(i)-forecast_horizon);
        ssrd_temp = MeteoInp.ssrd(tf);
        tot_rad(i,1) = sum(ssrd_temp); % [W/m2]
        clear ssrd_temp daily_step d td
        %
        mov_airt = MeteoInp.t2m(tf);
        airt(i,1) = mean(mov_airt);
        %
        totp_temp = MeteoInp.tp(tf);
        tprec(i,1) = sum(totp_temp); % [m]
        clear ssrd_temp daily_step d td        
        ws = MeteoInp.windspeed(tf);
        [f,x] = ecdf(ws);
        if ~isempty(find(x>3,1))
            Wind(i,1) = 1-f(find(x>3,1));
        else
            Wind(i,1) = 0;
        end
    end
        %% Create the dataset
        predictors = table();
        % radiation
        predictors.trad = tot_rad; %1
        % air temp
        predictors.airt = airt;%2
        % total precipitation	
        predictors.tprec = tprec;    %3
        % wind
        predictors.Wind = Wind; %4      
        % nutrients
        predictors.tn = (tn); %6
        predictors.tp = (tp);%7
        %
        target = Observations.value;
        predictors = table2array(predictors);
end
    
end

function [PredictorList] = FeatureSelection(predictors,target)
    rng(1);  
    %% create the cross-validation partitions
    c = cvpartition(length(predictors),'k',5); % 10-fold cross-validation 
    opts = statset('Display','iter','TolFun',0.01,'TolTypeFun','rel'); %
    %%
    fun = @(XT,yT,Xt,yt)loss(fitrensemble(XT,yT,'OptimizeHyperparameters','all', ...
        'HyperparameterOptimizationOptions',struct('ShowPlots',false,'Verbose',0,...
        'AcquisitionFunctionName','expected-improvement-plus')),Xt,yt);
    %%
    [fs,~] = sequentialfs(fun,normalize(predictors),normalize(target),'cv',c,'direction','backward','options',opts);
    PredictorList = 1:size(predictors,2);
    PredictorList = PredictorList(fs);
end

function [MdlRF,bestHyperparameters,Importance] = TrainRF(predictors,target,PredictorList)
    prd = predictors(:,PredictorList);
    minMinLS = max(5,round(size(prd,1)/40)); % 5 after breiman
    maxMinLS = round(size(prd,1)/10);
    if size(prd,2)<4
        minnumPTS = 1;
        maxnumPTS = 3;
    else
        minnumPTS = round(sqrt(size(prd,2))); % for low dimensions: Random forests: Some methodological insights. ArXiv preprint arXiv:0811.3619. URL: https://arxiv.org/abs/0811.3619.
        maxnumPTS = size(prd,2)-1; %
    end
    maxnumsplits = length(target)-1;
    minLS = optimizableVariable('minLS',[minMinLS,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[minnumPTS,maxnumPTS],'Type','integer');
    NumSplits = optimizableVariable('NumSplits',[1,maxnumsplits],'Type','integer');
    numTrees = optimizableVariable('numTrees',[20,500],'Type','integer');
    hyperparametersRF = [minLS;numPTS;NumSplits;numTrees];
    %
    results = bayesopt(@(params)ObjFun(params,prd,target),...
                hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',30,...
                'Verbose',0,'UseParallel',true,'PlotFcn',[]);
    bestHyperparameters = results.XAtMinObjective;
   %%
    MdlRF= TreeBagger(bestHyperparameters.numTrees,prd,...
       target,'Method','regression',...
        'OOBPrediction','on','OOBPredictorImportance','on','Surrogate','on',...
        'MinLeafSize',bestHyperparameters.minLS,...
        'NumPredictorstoSample',bestHyperparameters.numPTS, ...
        'PredictorSelection', 'interaction-curvature',...
        'MaxNumSplits',bestHyperparameters.NumSplits);
    Importance = MdlRF.OOBPermutedPredictorDeltaError;
end
function lossfun = ObjFun(params,predictors,target)
    %% 
    randomForest = TreeBagger(params.numTrees,predictors,...
    target,'Method','regression','Surrogate','on',...
    'OOBPrediction','on','OOBPredictorImportance','on','PredictorSelection', 'interaction-curvature',...
    'MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS,...
    'MaxNumSplits',params.NumSplits);
     lossfun = oobError(randomForest,'Mode','Ensemble','Quantile',0.9);
end

function [MdlQRF,bestHyperparameters] = TrainQRF(predictors,target,PredictorList)
    %
    prd = predictors(:,PredictorList);
    minMinLS = max(5,round(size(prd,1)/40)); % 5 after breiman
    maxMinLS = round(size(prd,1)/10);
    minnumPTS = round(sqrt(size(prd,2))); % for low dimensions: Random forests: Some methodological insights. ArXiv preprint arXiv:0811.3619. URL: https://arxiv.org/abs/0811.3619.
    maxnumPTS = size(prd,2)-1; %
    maxnumsplits = round(size(predictors,1)/3);
    minLS = optimizableVariable('minLS',[minMinLS,maxMinLS],'Type','integer');
    numPTS = optimizableVariable('numPTS',[minnumPTS,maxnumPTS],'Type','integer');
    NumSplits = optimizableVariable('NumSplits',[1,maxnumsplits],'Type','integer');
    numTrees = optimizableVariable('numTrees',[50,1000],'Type','integer');
    hyperparametersRF = [minLS;numPTS;NumSplits;numTrees];
    %%
    results = bayesopt(@(params)ObjFunQRF(params,prd,target),...
                hyperparametersRF,'AcquisitionFunctionName','expected-improvement-plus','MaxObjectiveEvaluations',100,...
                'Verbose',0,'UseParallel',true,'PlotFcn',[]);
    bestHyperparameters = results.XAtMinObjective;
    MdlQRF = TreeBagger(bestHyperparameters.numTrees,prd,...
        trg,'Method','regression',...
        'OOBPrediction','on','OOBPredictorImportance','on','Surrogate','on',...
        'MinLeafSize',bestHyperparameters.minLS,...
        'NumPredictorstoSample',bestHyperparameters.numPTS, ...
        'PredictorSelection', 'interaction-curvature',...
        'MaxNumSplits',bestHyperparameters.NumSplits);
end
function lossfun = ObjFunQRF(params,predictors,target)
    %% 
    randomForest = TreeBagger(params.numTrees,predictors,...
    target,'Method','regression','Surrogate','on',...
    'OOBPrediction','on','OOBPredictorImportance','on','PredictorSelection', 'interaction-curvature',...
    'MinLeafSize',params.minLS,'NumPredictorstoSample',params.numPTS,...
    'MaxNumSplits',params.NumSplits);
     lossfun = oobQuantileError(randomForest,'Mode','Ensemble','Quantile',0.9);
end

function [crps,mean_crps] = Crps(mdl,predictors,target)
    percentiles = [0:0.05:1];
    crps = zeros(length(target),1);
    %%
    
    for i =1:length(target)
        [ECDFobs,xobs] = ecdf(target(i));
        
        for j=1:length(percentiles)
            fval(j,1) = quantilePredict(mdl,predictors(i,:),'Quantile',percentiles(j));
        end
        [ECDFpred,xpred] = ecdf(fval);
        for j=1:length(ECDFpred)
            if xpred(j)<xobs(1)
                ecdfval = ECDFobs(1,1);
            else
                 ecdfval = ECDFobs(end,1);
            end
            if ECDFpred(j,1)>ecdfval
                indicator = 1;
            else
                indicator = 0;
            end
            crps_inner(j,1) = (ECDFpred(j,1) - indicator).^2;
        end
        crps(i,1)= trapz(xpred,crps_inner);
        clear crps_inner ECDFpred  xpred
    end
    mean_crps = mean(crps);
end

function [RS1,AW1,RS2,AW2] = EvalUnc(Model1,Model2,predictors,target,Quantile)
    q1 = quantilePredict(Model1,predictors,'Quantile',Quantile);
    RS1 = (length(find(q1(:,1)<target&q1(:,2)>target))/length(predictors))*100;
    AW1 = mean((q1(:,2)-q1(:,1)));
    
    q2 = quantilePredict(Model2,predictors,'Quantile',Quantile);
    RS2 = (length(find(q2(:,1)<target&q2(:,2)>target))/length(predictors))*100;
    AW2 = mean((q2(:,2)-q2(:,1)));
end