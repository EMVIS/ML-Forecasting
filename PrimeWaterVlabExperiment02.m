%% Step 01 -- Read chl-a from Geotiff providing the directory they reside and the coordinates of a point of interest
Observations = ReadGeoTiff(DirectoryName,PointOfInterest);
%% Step 02 -- PreProcess time-series data of satellite-derived chl-a
Observations = FillnSmooth(Observations); % Observations are used as timetables
%% Step 03 -- Find optimal width of the sliding windows for the prediction strategy
[window] = OptimalWindow(CaseStudy,HydroInp,MeteoInp,Observations,point,forecast_horizon);
% create final training dataset
[predictors,target] = CreateTrainingData(CaseStudy,HydroInp,MeteoInp,Observations,point,window,forecast_horizon);
[norm_data,C,S] = normalize([predictors target]);
%% Step 04 -- Feature Selection
PredictorList = FeatureSelection(norm_data(:,1:end-1),norm_data(:,end),ModelType);
%% Step 05 -- Model training using k-fold cross-validation
[rf,bestHyperparameters,~] = TrainRF(norm_data(:,1:end-1),norm_data(:,end),PredictorList);
%% Step 06 - ReForecast Experiment using expired forecasts of hydrological and meteorological data
ReForecastRun(Dates,Lake,ModelType,HydroInp,MeteoInp);
Output = ReadForecasts(ModelType);
%% Step 07 - Forecast Evaluation
[MASE] = mase_benchmark(Observations.time,Observations.value,tfrcst,frcst);

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
%%
function Observations = FillnSmooth(Observations)
	% Fill missing data
	Observations = fillmissing(Observations,"makima","MaxGap",days(3),...
	    "DataVariables","value");
	% Smooth input data
	Observations = smoothdata(Observations,"sgolay",days(7),"Degree",2,...
	    "DataVariables","value");
end
%%
function [window] = OptimalWindow(CaseStudy,HydroInp,MeteoInp,Observations,forecast_horizon)
    %%Estimate optimal window
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

            clear Window predictors target results
        end
    end 
    minimum = min(min(loss));
    [w1,w2]=find(loss==minimum);
    window = [Window1(w1,w2),Window2(w1,w2)];
end
%%
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
        % nutrients
        predictors.tn01 = (tn); %5
        predictors.tp01 = (tp);%6
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
%%
function [PredictorList] = FeatureSelection(predictors,target)
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
%%
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
%%
function lossfun = ObjFun(params,predictors,target)
    %%  
    ypred = zeros(length(target),5);
    for i=1:5
        cvp = cvpartition(length(target),'KFold',5);
        tTr = templateTree('PredictorSelection','interaction-curvature','Surrogate','on', ...
        'Reproducible','on','MaxNumSplits',params.NumSplits,...
        'MinLeafSize',params.minLS,'NumVariablesToSample',params.numPTS);
        randomForest = fitrensemble(predictors,target,'Method','Bag',...
        'NumLearningCycles',params.numTrees,...
        'Learners',tTr,'CrossVal','on','CVPartition',cvp);
        ypred(:,i) = kfoldPredict(randomForest);
    end
    lossfun = mean(abs(target-mean(ypred,2)));
end
%%
function ReForecastRun(Dates,Lake,HydroInp,MeteoInp)
    ModelPath = cd;
    Predictors = createPred(Lake,Dates(d),HydroInp,MeteoInp,ModelPath);
    %%
    try
        fprintf('Forecasting chlorophyll-a...\n');
        echo off
        if ~isempty(Predictors)
            [CentralValue,~] = Forecast(Lake,Predictors,ModelPath);
        else
            CentralValue = -9999;
            QuantilePrediction = -9999;
        end
    catch err
       fid = fopen('ErrorFile.txt','a+');
       fprintf(fid, '%s', err.getReport('extended', 'hyperlinks','off'));
       fclose(fid);
    end
    %%
    try
        fprintf('Creating JSON for water quality forecasts...\n');
        echo off
        if ~isempty(Predictors)
            [tfrcst,PointForecast,ProbForecast] = createOutput(CentralValue,QuantilePrediction,Dates(d));
            save(strcat('CHL_',ModelType,datestr(Dates(d),'yyyymmdd')),'tfrcst','PointForecast','ProbForecast');
        else
            ForecastOut = 'No forecast available';
            save(strcat('CHL_',ModelType,datestr(Dates(d),'yyyymmdd')),'ForecastOut');
        end
    catch err
       fid = fopen('ErrorFile.txt','a+');
       fprintf(fid, '%s', err.getReport('extended', 'hyperlinks','off'));
       fclose(fid);
    end

end
%%
function Predictors = createPred(Lake,Dates,HydroInp,MeteoInp,ModelPath)
    if contains(Lake,'us-harsha') == 1
        cd(ModelPath)
        model_file = dir('rf.mat');
        load(model_file.name)
        %%
        hydrotemp = HydroInp(HydroInp.issued_date>Dates-15 & HydroInp.issued_date <= Dates,:);
        [~,index] = unique(hydrotemp.target_date,'last');
        hydrotemp = hydrotemp(index,:);
        clear index
        %%
        meteotemp = MeteoInp(MeteoInp.issued_date>(Dates-15) & MeteoInp.issued_date <=Dates,:);
        [~,index] = unique(meteotemp.target_date,'last');
        meteotemp = meteotemp(index,:);
        for i=1:10
            dateHydro = datetime(Dates,'InputFormat','dd-MM-yyyy')+i ;
            dateMeteo = datetime(Dates,'InputFormat','dd-MM-yyyy')+i;
            % ...
            mov_cout = hydrotemp.cout01(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctn = hydrotemp.cctn01(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctp = hydrotemp.cctp01(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cout02 = hydrotemp.cout02(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctn02 = hydrotemp.cctn02(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctp02 = hydrotemp.cctp02(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            % ...           
            TN(i,1) = sum((mov_cctn.*mov_cout)*86400*1e-6)+sum((mov_cctn02.*mov_cout02)*86400*1e-6);  % tot kg
            TP(i,1) = sum((mov_cctp.*mov_cout)*86400*1e-6)+sum((mov_cctp02.*mov_cout02)*86400*1e-6);  % tot kg
            %% ...
            mov_airt = meteotemp.AirTemp(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            airt(i,1) = mean(mov_airt);
            % ...
            mov_rad = meteotemp.Radiation(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            trad(i,1) =sum((mov_rad));
            % ...
            mov_prec = meteotemp.Precipitation(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            tprec(i,1) = sum(mov_prec);
            %
            mov_windSpeed = meteotemp.WindSpeed(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            [f,x] = ecdf(mov_windSpeed);
            if ~isempty(find(x>3,1))
                Wind(i,1) = 1-f(find(x>3,1));
            else
                Wind(i,1) = 0;
            end
    
        end
        %%
        Predictors = table('Size',[10 7],'VariableTypes',{'datetime','double','double','double','double',...
            'double','double'});
        Predictors.Properties.VariableNames{1} = 'target_date';
        Predictors.Properties.VariableNames{2} = 'tot_rad';
        Predictors.Properties.VariableNames{3} = 'airt';
        Predictors.Properties.VariableNames{4} = 'tprec';
        Predictors.Properties.VariableNames{5} = 'Wind';
        Predictors.Properties.VariableNames{6} = 'TN';
        Predictors.Properties.VariableNames{7} = 'TP';
    
        Predictors.tot_rad = trad;
        Predictors.airt = airt;
        Predictors.tprec = tprec;
        Predictors.Wind= Wind;
        Predictors.TN = TN;
        Predictors.TP = TP;
        for i =1:10
            Predictors.target_date(i) =  datetime(Dates,'InputFormat','dd-MM-yyyy')+i;
        end
    elseif contains(Lake,'it-sardinia') == 1
        cd(ModelPath)
        model_file = dir('rf.mat');
        load(model_file.name)
        %%
        hydrotemp = HydroInp(HydroInp.issued_date>Dates-15 & HydroInp.issued_date <= Dates,:);
        [~,index] = unique(hydrotemp.target_date,'last');
        hydrotemp = hydrotemp(index,:);
        clear index
        %%
        meteotemp = MeteoInp(MeteoInp.issued_date>(Dates-15) & MeteoInp.issued_date <=Dates,:);
        [~,index] = unique(meteotemp.target_date,'last');
        meteotemp = meteotemp(index,:);
        for i=1:10
            dateHydro = datetime(Dates,'InputFormat','dd-MM-yyyy')+i ;
            dateMeteo = datetime(Dates,'InputFormat','dd-MM-yyyy')+i;
            % ...
            mov_cout = hydrotemp.cout(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctn = hydrotemp.cctn(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
            mov_cctp= hydrotemp.cctp(hydrotemp.target_date>dateHydro-14 & hydrotemp.target_date <=dateHydro);
             % ...           
            TN(i,1) = sum((mov_cctn.*mov_cout)*86400*1e-6);  % 
            TP(i,1) = sum((mov_cctp.*mov_cout)*86400*1e-6);  % 
            %% ...
            mov_airt = meteotemp.AirTemp(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            airt(i,1) = mean(mov_airt);
            % ...
            mov_rad = meteotemp.Radiation(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            trad(i,1) =sum((mov_rad));
            % ...
            mov_prec = meteotemp.Precipitation(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            tprec(i,1) = sum(mov_prec);
            %
            mov_windSpeed = meteotemp.WindSpeed(meteotemp.target_date>dateMeteo-14 & meteotemp.target_date <= dateMeteo);
            [f,x] = ecdf(mov_windSpeed);
            if ~isempty(find(x>3,1))
                Wind(i,1) = 1-f(find(x>3,1));
            else
                Wind(i,1) = 0;
            end
    
        end
        %%
        Predictors = table('Size',[10 7],'VariableTypes',{'datetime','double','double','double','double',...
            'double','double'});
        Predictors.Properties.VariableNames{1} = 'target_date';
        Predictors.Properties.VariableNames{2} = 'tot_rad';
        Predictors.Properties.VariableNames{3} = 'airt';
        Predictors.Properties.VariableNames{4} = 'tprec';
        Predictors.Properties.VariableNames{5} = 'Wind';
        Predictors.Properties.VariableNames{6} = 'TN';
        Predictors.Properties.VariableNames{7} = 'TP';
    
        Predictors.tot_rad = trad;
        Predictors.airt = airt;
        Predictors.tprec =tprec;
        Predictors.Wind= Wind;
        Predictors.TN = TN;
        Predictors.TP = TP;
        for i =1:10
            Predictors.target_date(i) =  datetime(Dates,'InputFormat','dd-MM-yyyy')+i;
        end
    else
            if point ==1 || point ==2
                %%
                cd(ModelPath)
                model_file = dir('rf.mat');
                load(model_file.name)
                W1 = GrowthWindow(1);
                W2 = GrowthWindow(2);
                hydrotemp = HydroInp(HydroInp.issued_date>Date-20 & HydroInp.issued_date <= Date,:);
                [~,index] = unique(hydrotemp.target_date,'last');
                hydrotemp = hydrotemp(index,:);
                clear index
                %%
                meteotemp = MeteoInp(MeteoInp.issued_date>(Date-30) & MeteoInp.issued_date <=Date,:);
                [~,index] = unique(meteotemp.target_date,'last');
                meteotemp = meteotemp(index,:);
                   
                for i=1:10
                    %%
                    dateHydro = datetime(Date,'InputFormat','dd-MM-yyyy') + i;
                    dateMeteo = datetime(Date,'InputFormat','dd-MM-yyyy')+ i;
                    % ...
                    mov_cout03 = hydrotemp.COUT03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn03 = hydrotemp.CCTN03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp03 = hydrotemp.CCTP03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    % ...
                    mov_cout13 = hydrotemp.COUT13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn13 = hydrotemp.CCTN13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp13 = hydrotemp.CCTP13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    % ...
                    TN13(i,1) = sum((mov_cctn13.*mov_cout13)*86400*1e-6);
                    TP13(i,1) = sum((mov_cctp13.*mov_cout13)*86400*1e-6);
                    TN03(i,1) = sum((mov_cctn03.*mov_cout03)*86400*1e-6);
                    TP03(i,1) = sum((mov_cctp03.*mov_cout03)*86400*1e-6);
                    %%    
                    mov_airt = meteotemp.AirTemp(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);             
                    airt(i,1) = mean(mov_airt);
                    % ...
                    mov_rad = meteotemp.Radiation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    trad(i,1) = sum(mov_rad);
                    % ...
                    mov_prec = meteotemp.Precipitation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    tprec(i,1) = sum(mov_prec);
                    %
                    mov_windSpeed = meteotemp.WindSpeed(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    [f,x] = ecdf(mov_windSpeed);
                    if ~isempty(find(x>3,1))
                        WindSpeed(i,1) = 1-f(find(x>3,1));  
                   else
                        WindSpeed(i,1)= 0;
                    end
                end
                %%
                Predictors = table('Size',[10 9],'VariableTypes',{'datetime','double','double','double','double',...
                    'double','double','double','double'});
                Predictors.Properties.VariableNames{1} = 'target_date';
                Predictors.Properties.VariableNames{2} = 'tot_rad';
                Predictors.Properties.VariableNames{3} = 'airt';
                Predictors.Properties.VariableNames{4} = 'tprec';
                Predictors.Properties.VariableNames{5} = 'WindSpeed';
                Predictors.Properties.VariableNames{6} = 'TN13';
                Predictors.Properties.VariableNames{7} = 'TN03';
                Predictors.Properties.VariableNames{8} = 'TP13';    
                Predictors.Properties.VariableNames{9} = 'TP03'; 
                %
                Predictors.tot_rad = trad;
                Predictors.airt = airt;
                Predictors.tprec = tprec;
                Predictors.WindSpeed = WindSpeed;
                Predictors.TN13 = TN13;
                Predictors.TN03 = TN03;
                Predictors.TP13 = TP13;
                Predictors.TP03= TP03;
                for i =1:10
                    Predictors.target_date(i) =  datetime(Date,'InputFormat','dd-MM-yyyy')+i;
                end
            elseif point ==3
                    cd(ModelPath)
                    model_file = dir('rf.mat');
                load(model_file.name)
                W1 = GrowthWindow(1);
                W2 = GrowthWindow(2);
                hydrotemp = HydroInp(HydroInp.issued_date>Date-20 & HydroInp.issued_date <= Date,:);
                [~,index] = unique(hydrotemp.target_date,'last');
                hydrotemp = hydrotemp(index,:);
                clear index
                %
                meteotemp = MeteoInp(MeteoInp.issued_date>(Date-30) & MeteoInp.issued_date <=Date,:);
                [~,index] = unique(meteotemp.target_date,'last');
                meteotemp = meteotemp(index,:);
                %%
                for i=1:10
                    %%
                    dateHydro = datetime(Date,'InputFormat','dd-MM-yyyy') + i;
                    dateMeteo = datetime(Date,'InputFormat','dd-MM-yyyy')+i;
                    % ...
                    mov_cout07 = hydrotemp.COUT07(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn07 = hydrotemp.CCTN07(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp07 = hydrotemp.CCTP07(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    % ...
                    mov_cout13 = hydrotemp.COUT13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn13 = hydrotemp.CCTN13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp13 = hydrotemp.CCTP13(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    % ...
                    mov_cout03 = hydrotemp.COUT03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn03 = hydrotemp.CCTN03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp03 = hydrotemp.CCTP03(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    %
                    TN13(i,1) = sum((mov_cctn13.*mov_cout13)*86400*1e-6);
                    TP13(i,1) = sum((mov_cctp13.*mov_cout13)*86400*1e-6);
                    TN07(i,1) = sum((mov_cctn07.*mov_cout07)*86400*1e-6);
                    TP07(i,1) = sum((mov_cctp07.*mov_cout07)*86400*1e-6);
                    TN03(i,1) = sum((mov_cctn03.*mov_cout03)*86400*1e-6);
                    TP03(i,1) = sum((mov_cctp03.*mov_cout03)*86400*1e-6);
                    %% -----
                    mov_airt = meteotemp.AirTemp(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);             
                    airt(i,1) = mean(mov_airt);
                    % ...
                    mov_rad = meteotemp.Radiation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    trad(i,1) = sum(mov_rad);
                    % ...
                    mov_prec = meteotemp.Precipitation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    tprec(i,1) = sum(mov_prec);
                    %
                    mov_windSpeed = meteotemp.WindSpeed(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    [f,x] = ecdf(mov_windSpeed);
                    if ~isempty(find(x>3,1))
                        WindSpeed(i,1) = 1-f(find(x>3,1));  
                   else
                        WindSpeed(i,1)= 0;
                    end
                end
                %%
                Predictors = table('Size',[10 11],'VariableTypes',{'datetime','double','double','double','double',...
                    'double','double','double','double','double','double'});
                Predictors.Properties.VariableNames{1} = 'target_date';
                Predictors.Properties.VariableNames{2} = 'tot_rad';
                Predictors.Properties.VariableNames{3} = 'airt';
                Predictors.Properties.VariableNames{4} = 'tprec';
                Predictors.Properties.VariableNames{5} = 'WindSpeed';
                Predictors.Properties.VariableNames{6} = 'TN07';
                Predictors.Properties.VariableNames{7} = 'TN13';
                Predictors.Properties.VariableNames{8} = 'TN03';
                Predictors.Properties.VariableNames{9} = 'TP07';    
                Predictors.Properties.VariableNames{10} = 'TP13'; 
                Predictors.Properties.VariableNames{11} = 'TP03';                       
                %
                Predictors.tot_rad = trad;
                Predictors.airt = airt;
                Predictors.tprec = tprec;
                Predictors.WindSpeed = WindSpeed;
                Predictors.TN07 = TN07;
                Predictors.TN13 = TN13;
                Predictors.TN03 = TN03;
                Predictors.TP07 = TP07;
                Predictors.TP13 = TP13;
                Predictors.TP03 = TP03;
                for i =1:10
                    Predictors.target_date(i) =  datetime(Date,'InputFormat','dd-MM-yyyy')+i;
                end
            else
                 %%
                cd(ModelPath)
                model_file = dir('rf.mat');
                load(model_file.name)
                W1 = GrowthWindow(1);
                W2 = GrowthWindow(2);
                hydrotemp = HydroInp(HydroInp.issued_date>Date-20 & HydroInp.issued_date <= Date,:);
                [~,index] = unique(hydrotemp.target_date,'last');
                hydrotemp = hydrotemp(index,:);
                clear index
                %
                meteotemp = MeteoInp(MeteoInp.issued_date>(Date-30) & MeteoInp.issued_date <=Date,:);
                [~,index] = unique(meteotemp.target_date,'last');
                meteotemp = meteotemp(index,:);
                %%
                for i=1:10
                    dateHydro = datetime(Date,'InputFormat','dd-MM-yyyy') + i;
                    dateMeteo = datetime(Date,'InputFormat','dd-MM-yyyy')+i;
                    % ...
                    mov_cout01= hydrotemp.COUT01(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctn01= hydrotemp.CCTN01(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    mov_cctp01= hydrotemp.CCTP01(hydrotemp.target_date>dateHydro-W1 & hydrotemp.target_date <=dateHydro);
                    TN01(i,1) = sum((mov_cctn01.*mov_cout01)*86400*1e-6);
                    TP01(i,1) = sum((mov_cctp01.*mov_cout01)*86400*1e-6);
                    %%    
                    mov_airt = meteotemp.AirTemp(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);             
                    airt(i,1) = mean(mov_airt);
                    % ...
                    mov_rad = meteotemp.Radiation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    trad(i,1) = sum(mov_rad);
                    % ...
                    mov_prec = meteotemp.Precipitation(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    tprec(i,1) = sum(mov_prec);
                    %
                    mov_windSpeed = meteotemp.WindSpeed(meteotemp.target_date>dateMeteo-W2 & meteotemp.target_date <= dateMeteo);
                    [f,x] = ecdf(mov_windSpeed);
                    if ~isempty(find(x>3,1))
                        WindSpeed(i,1) = 1-f(find(x>3,1));  
                   else
                        WindSpeed(i,1)= 0;
                    end            
                    %
                end
                %%
                Predictors = table('Size',[10 7],'VariableTypes',{'datetime','double','double','double','double',...
                    'double','double'});
                Predictors.Properties.VariableNames{1} = 'target_date';
                Predictors.Properties.VariableNames{2} = 'tot_rad';
                Predictors.Properties.VariableNames{3} = 'airt';
                Predictors.Properties.VariableNames{4} = 'tprec';
                Predictors.Properties.VariableNames{5} = 'WindSpeed';
                Predictors.Properties.VariableNames{6} = 'TN01';
                Predictors.Properties.VariableNames{7} = 'TP01';     
                %
                Predictors.tot_rad = trad;
                Predictors.airt = airt;
                Predictors.tprec = tprec;
                Predictors.WindSpeed = WindSpeed;
                Predictors.TN01= TN01;
                Predictors.TP01= TP01;
                for i =1:10
                    Predictors.target_date(i) =  datetime(Date,'InputFormat','dd-MM-yyyy')+i;
                end
            end
    
    end
end
%%
function [CentralValue,QuantilePrediction] = Forecast(Lake,Predictors,ModelPath)%% 
if contains(Lake,'us-harsha') == 1
        cd(ModelPath)
        model_file = dir('rf.mat');
        load(model_file.name)
        %%
        prd = table2array(Predictors(:,2:end));
        prd = normalize(prd,"center",C(1:end-1),"scale",S(1:end-1));
        prd = prd(:,prdlst);
        %%
        CentralValue(:,1) = predict(rf,prd).*S(end)+C(end);
        %%
        Quantile = [0.025:0.025:0.975];
        % 
        QuantilePrediction = quantilePredict(rf,prd,'Quantile',Quantile).*S(end)+C(end);
elseif contains(Lake,'it-sardinia') == 1
        cd(ModelPath)
        model_file = dir('rf.mat');
        load(model_file.name)
        %%
        prd = table2array(Predictors(:,2:end));
        prd = normalize(prd,"center",C(1:end-1),"scale",S(1:end-1));
        prd = prd(:,prdlst);
        %%
        CentralValue(:,1) = predict(rf,prd).*S(end)+C(end);
        %%
        Quantile = [0.025:0.025:0.975];
        % 
        QuantilePrediction = quantilePredict(rf,prd,'Quantile',Quantile).*S(end)+C(end);
else
        cd(ModelPath)
        model_file = dir('rf.mat');
        load(model_file.name)
        %%
        prd = table2array(Predictors(:,2:end));
        prd = normalize(prd,"center",C(1:end-1),"scale",S(1:end-1));
        prd = prd(:,prdlst);
        %%
        CentralValue(:,1) = predict(rf,prd).*S(end)+C(end);
        %%
        Quantile = [0.025:0.025:0.975];
        % 
        QuantilePrediction = quantilePredict(rf,prd,'Quantile',Quantile).*S(end)+C(end);
end
end
%%
function  [tfrcst,PointForecast,ProbForecast] = createOutput(CentralValue,QuantilePrediction,dt)
    tfrcst = datetime(dt,'InputFormat','dd-MM-yyyy','Format','yyyy-MM-dd''T''HH:mm:ss');
    for i=1:10
            PointForecast(i,1) = CentralValue(i);
            ProbForecast(i,1) = QuantilePrediction(i,38); 
            ProbForecast(i,2)= QuantilePrediction(i,2);
            ProbForecast(i,3) = QuantilePrediction(i,36);
            ProbForecast(i,4) = QuantilePrediction(i,4);
    end
end
%%    
function Output = ReadForecasts(DataPath)
    ForecastingHorizon = 10;
    cd(DataPath)
    list = dir('*CHL_rf*.mat');
    T = [datetime(2015,01,15):datetime(2018,12,31)]';
    PointForecast = zeros(length(list),ForecastingHorizon);
    %%
    for i=1:length(list)
        load(list(i).name)
        if isstruct(ForecastOut)
            for j=1:ForecastingHorizon
                PointForecast(i,j) = ForecastOut.MyMapSnapshots(j).Values(1);
            end
        else
            PointForecast(i,1:ForecastingHorizon) = NaN;
        end
    end
    %
    PointForecast(PointForecast<0) = NaN;
    Output = table(T,PointForecast);
    Output = sortrows(Output,'T','ascend');
end
%%
function [Mase] = mase_benchmark(tobs,Obs,tfrcst,frcst)
    %%
    Point01Naive = interp1(tobs,Obs,(tobs(1):1:tobs(end,1)),'previous')';
    Point01Naive = repmat(Point01Naive,[1,10]);
    t_naive = (tobs(1)+1:1:tobs(end,1)+1)';
    for i =1:length(t_naive)
        for j=1:10
            t_naive(i,j) = t_naive(i,1)+j;
        end
    end
    %%
    Mase = zeros(10,1);
    MAEn= zeros(10,1);
    MAEf= zeros(10,1);
    frcst(frcst ==-9999)=NaN;
    for i=1:10
        %
        [~,io,in] = intersect((tobs(~isnan(Obs))),t_naive(:,i));
        ErrNaive = abs(Obs(io,1) - Point01Naive(in,i));
        MAEn(i,1) = mean(ErrNaive(~isnan(ErrNaive)));
        clear io in
        %
        [~,io,is] = intersect(tobs,tfrcst(:,i));
        ErrDay = abs(Obs(io,1) - frcst(is,i));
        MAEf(i,1) = mean(ErrDay(~isnan(ErrDay)));
        %
        Mase(i,1) = MAEf(i,1)/MAEn(i,1);
        clear io is
    end
end