function bench_cracks(opts)
%% benchmarks for each method
if ~isfield('nthresh',opts)
    opts.nthresh = 99;
end
imgDir = '../data/images/test';
gtDir = '../data/groundTruth/test';
switch opts.method
    
    case 'crackIT'
        %% bench crackIT-original (comparison)
        pbDir = '../results/CrackIT/reshaped';
        outDir = '../eval/CrackIT';
        
    case 'CrackTree'
        %% bench crackTree (comparison)
        pbDir = '../results/CrackTree';
        outDir = '../eval/CrackTree';

    case 'SketchToken'
        %% bench SketchToken pre-trained model (comparison)
        pbDir = '../results/SketchToken';
        outDir = '../eval/SketchToken';
        
    case 'StructuredForest'
        %% bench StructuredForest pre-trained model (comparison)
        pbDir = '../results/StructuredForest';
        outDir = '../eval/StructuredForest';
        
    otherwise 
        %% bench crack_tokens (target)
        pbDir = fullfile(opts.outPath,'pmap');
        outDir = fullfile('../eval', 'crackToken', [opts.modelFnm '-' num2str(opts.reduceThr)]);
end
if ~exist(outDir,'dir'), mkdir(outDir); end
nthresh = opts.nthresh;
tic;
boundaryBench(imgDir, gtDir, pbDir, outDir, nthresh);
toc;
plot_eval(outDir);
end
