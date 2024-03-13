% % 此函数为最新一版
mdl = 'VSG1_RL';
open_system(mdl);
% Jmax = 0.45;
% Jmin = 0.05;
Dmax = 50;
Dmin = 20;
load("xFinal.mat");
load("Am.mat");
blk = 'VSG1_RL/RL Agent';
numObs = 4;
numAct = 2;
obsInfo = rlNumericSpec([numObs 1]);
obsInfo.Name = 'observations';
actInfo = rlNumericSpec([numAct 1],...
    'LowerLimit',[-1 -1]','UpperLimit',[1 1]');
actInfo.Name = 'JandD';
env = rlSimulinkEnv(mdl,blk,obsInfo,actInfo);
Ts0 = 0.002;
Tf0 = 0.8;
T0 = 0.002;
w0 = 100*pi;
rng(0)
%% Creat Agent
% observation input and path
cnet = [
    featureInputLayer(numObs,"Normalization","none","Name","observation")
    fullyConnectedLayer(128,"Name","fc1")
    concatenationLayer(1,2,"Name","concat")
    reluLayer("Name","relu1")
    fullyConnectedLayer(64,"Name","fc3")
    reluLayer("Name","relu2")
    fullyConnectedLayer(32,"Name","fc4")
    reluLayer("Name","relu3")
    fullyConnectedLayer(1,"Name","CriticOutput")];
actionPath = [
    featureInputLayer(numAct,"Normalization","none","Name","action")
    fullyConnectedLayer(128,"Name","fc2")];

% Connect the layers.
criticNetwork = layerGraph(cnet);
criticNetwork = addLayers(criticNetwork, actionPath);
criticNetwork = connectLayers(criticNetwork,"fc2","concat/in2");

criticdlnet = dlnetwork(criticNetwork,'Initialize',false);
criticdlnet1 = initialize(criticdlnet);
criticdlnet2 = initialize(criticdlnet);

critic1 = rlQValueFunction(criticdlnet1,obsInfo,actInfo, ...
    "ObservationInputNames","observation");
critic2 = rlQValueFunction(criticdlnet2,obsInfo,actInfo, ...
    "ObservationInputNames","observation");


% Create the actor network layers.
anet = [
    featureInputLayer(numObs,"Normalization","none","Name","observation")
    fullyConnectedLayer(128,"Name","fc1")
    reluLayer("Name","relu1")
    fullyConnectedLayer(64,"Name","fc2")
    reluLayer("Name","relu2")];
meanPath = [
    fullyConnectedLayer(32,"Name","meanFC")
    reluLayer("Name","relu3")
    fullyConnectedLayer(numAct,"Name","mean")];
stdPath = [
    fullyConnectedLayer(numAct,"Name","stdFC")
    reluLayer("Name","relu4")
    softplusLayer("Name","std")];

% Connect the layers.
actorNetwork = layerGraph(anet);
actorNetwork = addLayers(actorNetwork,meanPath);
actorNetwork = addLayers(actorNetwork,stdPath);
actorNetwork = connectLayers(actorNetwork,"relu2","meanFC/in");
actorNetwork = connectLayers(actorNetwork,"relu2","stdFC/in");

actordlnet = dlnetwork(actorNetwork);
actor = rlContinuousGaussianActor(actordlnet, obsInfo, actInfo, ...
    "ObservationInputNames","observation", ...
    "ActionMeanOutputNames","mean", ...
    "ActionStandardDeviationOutputNames","std");

agentOpts = rlSACAgentOptions( ...
    "SampleTime",Ts0, ...
    "TargetSmoothFactor",5e-3, ...    
    "ExperienceBufferLength",10e4, ...
    "MiniBatchSize",128, ...
    "NumWarmStartSteps",1000, ...
    "DiscountFactor",0.99);

agentOpts.ActorOptimizerOptions.Algorithm = "adam";
agentOpts.ActorOptimizerOptions.LearnRate = 1e-3;
agentOpts.ActorOptimizerOptions.GradientThreshold = 1;

for ct = 1:2
    agentOpts.CriticOptimizerOptions(ct).Algorithm = "adam";
    agentOpts.CriticOptimizerOptions(ct).LearnRate = 1e-3;
    agentOpts.CriticOptimizerOptions(ct).GradientThreshold = 1;
end
agent = rlSACAgent(actor,[critic1,critic2],agentOpts);

%% train agent
trainOpts = rlTrainingOptions(...
    "MaxEpisodes", 1000, ...
    "MaxStepsPerEpisode", floor(Tf0/Ts0), ...
    "ScoreAveragingWindowLength", 100, ...
    "Plots", "training-progress", ...
    "StopTrainingCriteria", "AverageReward", ...
    "StopTrainingValue", 675, ...
    "UseParallel", false);

if trainOpts.UseParallel
    trainOpts.ParallelizationOptions.AttachedFiles = [pwd,filesep] + ...
        ["bracelet_with_vision_link.STL";
        "half_arm_2_link.STL";
        "end_effector_link.STL";
        "shoulder_link.STL";
        "base_link.STL";
        "forearm_link.STL";
        "spherical_wrist_1_link.STL";
        "bracelet_no_vision_link.STL";
        "half_arm_1_link.STL";
        "spherical_wrist_2_link.STL"];
end

doTraining = 1;
if doTraining
    stats = train(agent,env,trainOpts);
end