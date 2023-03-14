    %% Step 01 -- Read Geotiff
    Observations = ReadGeoTiff(DirectoryName,PointOfInterest);
    %% Step 02 -- PreProcess Chl Data
    Observations = FillnSmooth(Observations);
    %% Step 03 -- Find optimal width of the sliding windows
    [window] = OptimalWindow(CaseStudy,HydroInp,MeteoInp,Observations,forecast_horizon);
    % create final training dataset
    [predictors,target] = CreateTrainingData(CaseStudy,HydroInp,MeteoInp,Observations,point,window,forecast_horizon);
    [norm_data,C,S] = normalize([predictors target]);
    %% Step 04 -- Feature Selection
	PredictorList = FeatureSelection(norm_data(:,1:end-1),norm_data(:,end));
    %% Step 05 -- Model training using k-fold cross-validation
    [rf,bestHyperparameters,importance] = TrainRF(norm_data(:,1:end-1),norm_data(:,end),PredictorList);
    outofbagError = oobError(rf,'Mode','Ensemble');
    % estimate cross-validation errors
    [CVpredictions] = CrossValErrors(bestHyperparameters,norm_data(:,1:end-1),norm_data(:,end));
    CVpredictions = CVpredictions.*S(end)+C(end);
    CVerrors = CVpredictions - target;
    %% Step 06 - Model Interpretation
    % Plot Individual Conditional Expectation Plots
    plotsize = length(PredictorList);
    figure
    t = tiledlayout(2,plotsize-2,"TileSpacing","compact");
    title(t,"Individual Conditional Expectation Plots")
    for i = 1 : plotsize
        nexttile
        plotPartialDependence(rf,i,norm_data(:,PredictorList),"Conditional","absolute")
        title("")
    end
    %% Estimate Shapley values
    Ypred = predict(rf, norm_data(:,PredictorList));
    tbl = norm_data(:,PredictorList);
    f = @(tbl) predict(rf,tbl);
    explainer = shapley(f,tbl);
    for i=1:length(Ypred)
        explainer = fit(explainer,tbl(i,:));
        ShapleyVal(:,i) = explainer.ShapleyValues.ShapleyValue;
    end
   
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

function Observations = FillnSmooth(Observations)
	% Fill missing data
	Observations = fillmissing(Observations,"makima","MaxGap",days(3),...
	    "DataVariables","value");
	% Smooth input data
	Observations = smoothdata(Observations,"sgolay",days(7),"Degree",2,...
	    "DataVariables","value");
end

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
        weights = Observations.weight;
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
        weights = Observations.weight;
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
        weights = Observations.weight;
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
        weights = Observations.weight;
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
        weights = Observations.weight;
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
    maxnumsplits = length(trg)-1;
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
       trg,'Method','regression',...
        'OOBPrediction','on','OOBPredictorImportance','on','Surrogate','on',...
        'MinLeafSize',bestHyperparameters.minLS,...
        'NumPredictorstoSample',bestHyperparameters.numPTS, ...
        'PredictorSelection', 'interaction-curvature',...
        'MaxNumSplits',bestHyperparameters.NumSplits);
    Importance = MdlRF.OOBPermutedPredictorDeltaError;
end

function [CVpredictions] = CrossValErrors(bestHyperparameters,predictors,target)
    cvp = cvpartition(length(target),'KFold',5);
    tTr = templateTree('PredictorSelection','interaction-curvature','Surrogate','on', ...
    'Reproducible','on','MaxNumSplits',bestHyperparameters.NumSplits,...
    'MinLeafSize',bestHyperparameters.minLS,'NumVariablesToSample',bestHyperparameters.numPTS);
    
    randomForest = fitrensemble(predictors,target,'Method','Bag',...
    'NumLearningCycles',bestHyperparameters.numTrees,...
    'Learners',tTr,'CrossVal','on','CVPartition',cvp);
    CVpredictions = kfoldPredict(randomForest);
end
