function ManipulabilityLearning
addpath('../../fcts/');

%% Parameters
nbData = 100; % Number of datapoints in a trajectory
nbSamples = 4; % Number of demonstrations
nbIter = 10; % Number of iteration for the Gauss Newton algorithm (Riemannian manifold)
nbIterEM = 10; % Number of iteration for the EM algorithm
letter = 'C'; % Letter to use as dataset for demonstration data

modelPD.nbStates = 5; %Number of Gaussians in the GMM over man. ellipsoids
modelPD.nbVar = 4; % Dimension of the manifold and tangent space (1D input + 2^2 output)
modelPD.nbVarOut = 2; % Dimension of the output
modelPD.nbVarOutVec = modelPD.nbVarOut + modelPD.nbVarOut*(modelPD.nbVarOut-1)/2; % Dimension of the output in vector form
modelPD.nbVarVec = modelPD.nbVar - modelPD.nbVarOut + modelPD.nbVarOutVec; % Dimension of the manifold and tangent space in vector form
modelPD.nbVarCovOut = modelPD.nbVar + modelPD.nbVar*(modelPD.nbVar-1)/2; %Dimension of the output covariance
modelPD.dt = 1E-2; % Time step duration
modelPD.params_diagRegFact = 1E-4; % Regularization of covariance
modelPD.Kp = 100; % Gain for position control in task space

modelKin.nbStates = 5; % Number of states in the GMM over 2D Cartesian trajectories
modelKin.nbVar = 3; % Number of variables [t,x1,x2]
modelKin.dt = modelPD.dt; % Time step duration

%% Create robots
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Robots parameters
nbDOFs = 3; % Nb of degrees of freedom for teacher robot
% armLength = 4; % For I and L
armLength = 5; % For C
L1 = Link('d', 0, 'a', armLength, 'alpha', 0);
robotT = SerialLink(repmat(L1,nbDOFs,1)); % Robot teacher
q0T = [pi/4 0.0 -pi/9]; % Initial robot configuration

%% Load handwriting data and generating manipulability ellipsoids
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Loading demonstration data...');
dataPath = fullfile(fileparts(mfilename('fullpath')), '..', '..', 'data', '2Dletters', [letter '.mat']);
load(dataPath);
%load(['../../data/2Dletters/' letter '.mat'])
% 1) 算出本 .m 文件所在的文件夹
rootFolder = fileparts(mfilename('fullpath'));  
% 2) 从它开始，拼到 data/2Dletters/C.mat
dataPath = fullfile(rootFolder, '..','..','data','2Dletters',[letter '.mat']);
% 3) 只加载 demos 这个变量，然后取出来
S = load(dataPath);  
demos = S.demos; 


xIn(1,:) = (1:nbData) * modelPD.dt; % Time as input variable
X = zeros(3,3,nbData*nbSamples); % Matrix storing t,x1,x2 for all the demos
X(1,1,:) = reshape(repmat(xIn,1,nbSamples),1,1,nbData*nbSamples); % Stores input
Data=[];

for n=1:nbSamples
    s(n).Data=[];
    dTmp = spline(1:size(demos{n}.pos,2), demos{n}.pos, linspace(1,size(demos{n}.pos,2),nbData)); %Resampling
    s(n).Data = [s(n).Data; dTmp];
    
    % Obtain robot configurations for the current demo given initial robot pose q0
    T = transl([s(n).Data(1:2,:) ; zeros(1,nbData)]');
    
    % One way to check robotics toolbox version
    if isobject(robotT.fkine(q0T))  % 10.X
        maskPlanarRbt = [ 1 1 0 0 0 0 ];  % Mask matrix for a 3-DoFs robots for position (x,y)
        q = robotT.ikine(T, q0T', 'mask', maskPlanarRbt)';  % Based on an initial pose
    else  % 9.X
        maskPlanarRbt = [ 1 1 1 0 0 0 ];
        q = robotT.ikine(T, q0T', maskPlanarRbt)'; % Based on an initial pose
    end
    s(n).q = q; % Storing joint values
    
    % Computing force/velocity manipulability ellipsoids, that will be later
    % used for encoding a GMM in the force/velocity manip. ellip. manifold
    for t = 1 : nbData
        auxJ = robotT.jacob0(q(:,t),'trans');
        J = auxJ(1:2,:);
        X(2:3,2:3,t+(n-1)*nbData) = J*J'; % Saving ME
    end
    Data = [Data [xIn ; s(n).Data]]; % Storing time and Cartesian positions
end
% SPD data in vector shape
x = [reshape(X(1,1,:),1,nbData*nbSamples); symmat2vec(X(2:end,2:end,:))];
x = [ Data(2,:); Data(3,:); x ];  %就是 x1 x2
x(3,:)=[]   
disp(n)


%% GMM parameters estimation
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('Learning GMM1 (2D Cartesian position)...');
%modelKin = init_GMM_timeBased(Data, modelKin); % Model for position
modelKin = init_GMM_stateBased(Data, modelKin); 
modelKin = EM_GMM(Data, modelKin);

disp('Learning GMM2 (Manipulability ellipsoids)...');
% Initialisation on the manifold
in=1:2; outMat=3:modelPD.nbVar; out = 3:modelPD.nbVarVec;
modelPD = spd_init_GMM_kbins(x, modelPD, nbSamples,out);
modelPD.Mu = zeros(size(modelPD.MuMan));
L = zeros(modelPD.nbStates, nbData*nbSamples);
xts = zeros(modelPD.nbVarVec, nbData*nbSamples, modelPD.nbStates);

% EM for SPD matrices manifold
for nb=1:nbIterEM
    fprintf('.');
    % E-step
    for i=1:modelPD.nbStates
        xts(in,:,i) = x(in,:)-repmat(modelPD.MuMan(in,i),1,nbData*nbSamples);
        xts(out,:,i) = logmap_vec(x(out,:), modelPD.MuMan(out,i));
        L(i,:) = modelPD.Priors(i) * gaussPDF(xts(:,:,i), modelPD.Mu(:,i), modelPD.Sigma(:,:,i));
    end
    % Responsibilities
    GAMMA = L ./ repmat(sum(L,1)+realmin, modelPD.nbStates, 1);
    H = GAMMA ./ repmat(sum(GAMMA,2)+realmin, 1, nbData*nbSamples);
    
    % M-step
    for i=1:modelPD.nbStates
        % Update Priors
        modelPD.Priors(i) = sum(GAMMA(i,:)) / (nbData*nbSamples);
        % Update MuMan
        for n=1:nbIter
            % Update on the tangent space
            uTmp = zeros(modelPD.nbVarVec,nbData*nbSamples);
            uTmp(in,:) = x(in,:) - repmat(modelPD.MuMan(in,i),1,nbData*nbSamples);
            uTmp(out,:) = logmap_vec(x(out,:), modelPD.MuMan(out,i));
            uTmpTot = sum(uTmp.*repmat(H(i,:),modelPD.nbVarVec,1),2);
            % Update on the manifold
            modelPD.MuMan(in,i) = uTmpTot(in,:) + modelPD.MuMan(in,i);
            modelPD.MuMan(out,i) = expmap_vec(uTmpTot(out,:), modelPD.MuMan(out,i));
        end
        % Update Sigma
        modelPD.Sigma(:,:,i) = uTmp * diag(H(i,:)) * uTmp' + eye(modelPD.nbVarVec) .* modelPD.params_diagRegFact;
    end
end
% Eigendecomposition of Sigma
for i=1:modelPD.nbStates
    [V(:,:,i), D(:,:,i)] = eig(modelPD.Sigma(:,:,i));
end
modelPD.V = V
modelPD.D = D

% Inputs
xIn = zeros(2,nbSamples*nbData);  % 2*400
xIn(1,:) =  Data(2,:); 
xIn(2,:) =  Data(3,:);
figure('position',[10 10 1200 550],'color',[1 1 1]);
subplot(2,2,1); hold on;

title('\fontsize{12}Manipulability field with input of 2d position');
for t=1:nbData
    a(:,t) = GMR_mani(xIn(:,t) , modelPD)
end

for t=1:5:nbData % Plotting estimated man. ellipsoid from GMR
    plotGMM(xIn(:,t), 5E-2*vec2symmat(a(:,t)), [0.2 0.8 0.2], .5, '-.', 2, 1); 
end
axis equal;
set(gca, 'FontSize', 20)
xlabel('$x_1$', 'Fontsize', 28, 'Interpreter', 'Latex');
ylabel('$x_2$', 'Fontsize', 28, 'Interpreter', 'Latex');

plot(Data(2,1:100),Data(3,1:100), 'color', [0.5 0.5 0.5], 'Linewidth', 2);

b = GMR_mani([0,0] , modelPD)
plotGMM([0;0], 5E-2*vec2symmat(b), [0.2 0.8 1], .5, '-.', 2, 1); 
c = GMR_mani([20,5] , modelPD)
plotGMM([20;5], 5E-2*vec2symmat(c), [0.2 0.8 1], .5, '-.', 2, 1); 
d = GMR_mani([-8,-5] , modelPD)
plotGMM([-8;-5], 5E-2*vec2symmat(d), [0.2 0.8 1], .5, '-.', 2, 1); 
