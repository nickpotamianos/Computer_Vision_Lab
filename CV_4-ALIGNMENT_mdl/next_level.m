function warp_out = next_level(warp_in, transform, direction)
% This function modifies warp parameters between pyramid levels
% direction=1 --> from coarse to fine level
% direction=0 --> from fine to coarse level

if direction==1
    if strcmp(transform,'affine')||strcmp(transform,'euclidean')||strcmp(transform,'translation')
        warp_out = warp_in;
        warp_out(1:2,3) = 2 * warp_out(1:2,3);
    else
        warp_out = warp_in;
        warp_out(:,3) = 2 * warp_out(:,3);
    end
else
    if strcmp(transform,'affine')||strcmp(transform,'euclidean')||strcmp(transform,'translation')
        warp_out = warp_in;
        warp_out(1:2,3) = warp_out(1:2,3)/2;
    else
        warp_out = warp_in;
        warp_out(:,3) = warp_out(:,3)/2;
    end
end