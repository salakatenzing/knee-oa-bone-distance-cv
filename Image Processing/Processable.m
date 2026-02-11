function bool = Processable(image, filename)
    count = zeros(1,2);
    
    count(1)=sum(image(:)==0);
    count(2)=sum(image(:)~=0);
    
    %disp(['Non-Zero Percentage:',num2str()])
    
    nonzero_percentage = (count(2)/numel(image))*100;
    
    %disp(nonzero_percentage);
    
    if nonzero_percentage>20
        BinaryImage = image;
        
        se=strel('square',5);
        BinaryImage=imerode(BinaryImage,se);
        BinaryImage(1:135,:)= 255;
    
        BinaryImage(1:90,:)= 255;

        % step 4
        % labels each of the component to a specfic label/value to identify it
        labeledOutputImage = logical(BinaryImage);
        regions = regionprops(labeledOutputImage, 'Area', 'PixelIdxList', 'BoundingBox');

        % step 5
        % idenfiying the femur and tibia
        [~, sortedIdx] = sort([regions.Area], 'descend');
        if length(sortedIdx) < 2
            %disp(['Warning: Not enough regions detected in image ',filename]);
            bool = false;
        else
            bool = true;
        end
    else
        bool = false;
    end
    
end
