clc; clear; close all;

% ============================================================
% Improved SWT–TSSF–DCS for COLOUR images
% Each RGB channel fused separately.
% ============================================================

% -------------------- Parameters --------------------
P.level   = 3;
P.wname   = 'db2';

% LF (TSSF)
P.winLF   = 7;
P.alphaSF = 0.6;
P.Tmin    = 0.35;
P.Tmax    = 1.35;

% Weight smoothing (edge-aware on weights, not on fused image)
P.useWeightGuided = true;
P.gfRadiusLF = 6;
P.gfEpsLF    = 1e-4;

% HF (DCS)
P.winHF   = 5;
P.p0      = 1.3;     % base p-sharpen
P.beta0   = 0.65;    % base sign penalty
P.Thf0    = 0.55;    % base temperature
P.gfRadiusHF = 4;
P.gfEpsHF    = 2e-4;

% -------------------- Load colour images --------------------
[file1, path1] = uigetfile({'*.*','All Files'}, 'Load Image 1');
if isequal(file1,0), error('No Image 1 selected.'); end
im1 = imread(fullfile(path1,file1));

[file2, path2] = uigetfile({'*.*','All Files'}, 'Load Image 2');
if isequal(file2,0), error('No Image 2 selected.'); end
im2 = imread(fullfile(path2,file2));

% Ensure both are RGB (if grayscale, replicate to 3 channels)
if size(im1,3)==1, im1 = repmat(im1,[1,1,3]); end
if size(im2,3)==1, im2 = repmat(im2,[1,1,3]); end
if ~isequal(size(im1), size(im2)), error('Images must be same size'); end

% Normalise to [0,1] and convert to single
I1n = im2single(mat2gray(im1));   % size M×N×3
I2n = im2single(mat2gray(im2));

% Preallocate fused image
F = zeros(size(I1n), 'like', I1n);

% ============================================================
% Process each colour channel independently
% ============================================================
for ch = 1:3
    % Extract channel
    I1_ch = I1n(:,:,ch);
    I2_ch = I2n(:,:,ch);
    
    % Guide image for this channel (only for smoothing weight maps)
    G = 0.5 * (I1_ch + I2_ch);
    
    % SWT decomposition
    [cA1, cH1, cV1, cD1] = swt2(I1_ch, P.level, P.wname);
    [cA2, cH2, cV2, cD2] = swt2(I2_ch, P.level, P.wname);
    
    fA = zeros(size(cA1), 'like', cA1);
    fH = zeros(size(cH1), 'like', cH1);
    fV = zeros(size(cV1), 'like', cV1);
    fD = zeros(size(cD1), 'like', cD1);
    
    % ---------- LF Fusion (TSSF) ----------
    for l = 1:P.level
        A1 = cA1(:,:,l);  A2 = cA2(:,:,l);
        
        v1  = local_var(A1, P.winLF);
        v2  = local_var(A2, P.winLF);
        sf1 = local_sf(A1, P.winLF);
        sf2 = local_sf(A2, P.winLF);
        
        s1 = v1 + P.alphaSF * sf1;
        s2 = v2 + P.alphaSF * sf2;
        
        contrast = sqrt(v1 + v2 + eps('single'));
        T = clamp(1 ./ (1 + 6*contrast), P.Tmin, P.Tmax);
        
        [w1, w2] = softmax2(s1, s2, T);
        
        if P.useWeightGuided
            w1 = guided_weight(w1, G, P.gfRadiusLF, P.gfEpsLF);
            w1 = clamp(w1, 0, 1);
            w2 = 1 - w1;
        end
        
        fA(:,:,l) = w1 .* A1 + w2 .* A2;
    end
    
    % ---------- HF Fusion (DCS) ----------
    for l = 1:P.level
        H1 = cH1(:,:,l); H2 = cH2(:,:,l);
        V1 = cV1(:,:,l); V2 = cV2(:,:,l);
        D1 = cD1(:,:,l); D2 = cD2(:,:,l);
        
        eH1 = local_energy(H1, P.winHF); eH2 = local_energy(H2, P.winHF);
        eV1 = local_energy(V1, P.winHF); eV2 = local_energy(V2, P.winHF);
        eD1 = local_energy(D1, P.winHF); eD2 = local_energy(D2, P.winHF);
        
        conflictH = imboxfilt(single((H1.*H2)<0), P.winHF);
        conflictV = imboxfilt(single((V1.*V2)<0), P.winHF);
        conflictD = imboxfilt(single((D1.*D2)<0), P.winHF);
        conflict  = clamp((conflictH + conflictV + conflictD)/3, 0, 1);
        
        beta = clamp(P.beta0 + 0.25*conflict, 0.40, 0.95);
        Thf  = clamp(P.Thf0  - 0.20*conflict, 0.25, 0.80);
        p    = clamp(P.p0    + 0.40*conflict, 1.00, 2.00);
        
        fH(:,:,l) = fuse_hf_band(H1, H2, eH1, eH2, p, beta, Thf, G, P);
        fV(:,:,l) = fuse_hf_band(V1, V2, eV1, eV2, p, beta, Thf, G, P);
        fD(:,:,l) = fuse_hf_band(D1, D2, eD1, eD2, p, beta, Thf, G, P);
    end
    
    % Reconstruct this channel
    F_ch = iswt2(fA, fH, fV, fD, P.wname);
    F(:,:,ch) = im2single(mat2gray(F_ch));
end

% ============================================================
% Display and save colour fused image
% ============================================================
figure('Color','w');
subplot(1,3,1); imshow(I1n); title('Image 1 (colour)');
subplot(1,3,2); imshow(I2n); title('Image 2 (colour)');
subplot(1,3,3); imshow(F);   title('Fused Colour Image');

imwrite(uint8(255*F), 'fused_colour_swt_tssf_dcs.png');

% ============================================================
% Compute metrics on GRAYSCALE version (standard practice)
% ============================================================
F_gray = rgb2gray(F);
I1_gray = rgb2gray(I1n);
I2_gray = rgb2gray(I2n);

best.QABF = qabf_metric(I1_gray, I2_gray, F_gray);
best.QY   = qy_metric(I1_gray, I2_gray, F_gray);
best.QCB  = qcb_metric(I1_gray, I2_gray, F_gray);
best.QS   = qs_metric(I1_gray, I2_gray, F_gray);

fprintf('\n=== REQUIRED METRICS (on grayscale) ===\n');
fprintf('QAB/F : %.4f\n', best.QABF);
fprintf('QY    : %.4f\n', best.QY);
fprintf('QCB   : %.4f\n', best.QCB);
fprintf('QS    : %.4f\n\n', best.QS);

% ============================================================
% ======================= Helper functions (unchanged) ====================
% ============================================================

function v = local_var(x, w)
    mu  = imboxfilt(x, w);
    mu2 = imboxfilt(x.^2, w);
    v = max(mu2 - mu.^2, 0);
end

function sf = local_sf(x, w)
    dx = [diff(x,1,2), zeros(size(x,1),1,'like',x)];
    dy = [diff(x,1,1); zeros(1,size(x,2),'like',x)];
    sf = imboxfilt(dx.^2 + dy.^2, w);
end

function e = local_energy(x, w)
    e = sqrt(imboxfilt(x.^2, w) + eps('single'));
end

function [w1, w2] = softmax2(s1, s2, T)
    m  = max(s1, s2);
    a1 = exp((s1 - m) ./ (T + eps('single')));
    a2 = exp((s2 - m) ./ (T + eps('single')));
    den = a1 + a2 + eps('single');
    w1 = a1 ./ den;
    w2 = a2 ./ den;
end

function y = clamp(x, lo, hi)
    y = min(max(x, lo), hi);
end

function w = guided_weight(w, guide, r, epsv)
    if exist('imguidedfilter','file') == 2
        w = imguidedfilter(w, guide, ...
            'NeighborhoodSize', 2*r+1, ...
            'DegreeOfSmoothing', epsv);
    else
        w = imboxfilt(w, 2*r+1);
    end
end

function f = fuse_hf_band(B1, B2, e1, e2, p, beta, T, G, P)
    s1 = e1.^p;
    s2 = e2.^p;
    
    [w1, w2] = softmax2(s1, s2, T);
    
    opp = (B1 .* B2) < 0;
    weaker1 = e1 < e2;
    weaker2 = ~weaker1;
    
    w1 = w1 .* (1 - beta .* single(opp & weaker1));
    w2 = w2 .* (1 - beta .* single(opp & weaker2));
    
    den = w1 + w2 + eps('single');
    w1  = w1 ./ den;
    w2  = w2 ./ den;
    
    if P.useWeightGuided
        w1 = guided_weight(w1, G, P.gfRadiusHF, P.gfEpsHF);
        w1 = clamp(w1, 0, 1);
        w2 = 1 - w1;
    end
    
    f = w1 .* B1 + w2 .* B2;
end

function Q = qabf_metric(A, B, F)
    A = im2single(A); B = im2single(B); F = im2single(F);
    
    sobelx = fspecial('sobel');
    sobely = sobelx';
    
    Ax = imfilter(A, sobelx, 'replicate'); Ay = imfilter(A, sobely, 'replicate');
    Bx = imfilter(B, sobelx, 'replicate'); By = imfilter(B, sobely, 'replicate');
    Fx = imfilter(F, sobelx, 'replicate'); Fy = imfilter(F, sobely, 'replicate');
    
    GA = hypot(Ax,Ay); GB = hypot(Bx,By); GF = hypot(Fx,Fy);
    OA = atan2(Ay, Ax); OB = atan2(By, Bx); OF = atan2(Fy, Fx);
    
    QgA = (2*GA.*GF + eps('single')) ./ (GA.^2 + GF.^2 + eps('single'));
    QgB = (2*GB.*GF + eps('single')) ./ (GB.^2 + GF.^2 + eps('single'));
    
    dOA = abs(angle(exp(1j*(OA-OF))));
    dOB = abs(angle(exp(1j*(OB-OF))));
    QoA = 1 - dOA/pi;
    QoB = 1 - dOB/pi;
    
    QA = QgA .* QoA;
    QB = QgB .* QoB;
    
    wA = GA ./ (GA + GB + eps('single'));
    wB = 1 - wA;
    
    Qmap = wA .* QA + wB .* QB;
    Q = mean(Qmap(:));
end

function QY = qy_metric(A, B, F)
    QY = qabf_metric(A, B, F);
end

function QCB = qcb_metric(A, B, F)
    A = im2single(A); B = im2single(B); F = im2single(F);
    
    w = 7;
    cA = sqrt(local_var(A, w) + eps('single'));
    cB = sqrt(local_var(B, w) + eps('single'));
    cF = sqrt(local_var(F, w) + eps('single'));
    
    sA = (2*cA.*cF + eps('single')) ./ (cA.^2 + cF.^2 + eps('single'));
    sB = (2*cB.*cF + eps('single')) ./ (cB.^2 + cF.^2 + eps('single'));
    
    wA = cA ./ (cA + cB + eps('single'));
    wB = 1 - wA;
    
    Qmap = wA .* sA + wB .* sB;
    QCB = mean(Qmap(:));
end

function QS = qs_metric(A, B, F)
    A = im2single(A); B = im2single(B); F = im2single(F);
    
    if exist('ssim','file') == 2
        [~, ssimA] = ssim(F, A);
        [~, ssimB] = ssim(F, B);
    else
        ssimA = local_corr(F, A, 7);
        ssimB = local_corr(F, B, 7);
    end
    
    sobelx = fspecial('sobel'); sobely = sobelx';
    GA = hypot(imfilter(A,sobelx,'replicate'), imfilter(A,sobely,'replicate'));
    GB = hypot(imfilter(B,sobelx,'replicate'), imfilter(B,sobely,'replicate'));
    wA = GA ./ (GA + GB + eps('single'));
    wB = 1 - wA;
    
    QS = mean((wA .* ssimA + wB .* ssimB), 'all');
end

function C = local_corr(X, Y, w)
    mx = imboxfilt(X, w); my = imboxfilt(Y, w);
    vx = imboxfilt(X.^2, w) - mx.^2;
    vy = imboxfilt(Y.^2, w) - my.^2;
    cxy = imboxfilt(X.*Y, w) - mx.*my;
    
    C = (cxy + eps('single')) ./ sqrt((vx + eps('single')).*(vy + eps('single')));
    C = max(min(C,1),-1);
end