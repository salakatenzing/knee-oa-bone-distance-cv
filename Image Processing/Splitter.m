function b = Splitter(a)

    a_size = size(a);

    chunk = floor(a_size(2)/3);

    b=zeros(3,1);

    start = 1;

    for i=1:3
        if i==3
            t=mean(a(1,start:numel(a)));
        else
            t=mean(a(1,start:i+chunk));
        end

        start=start+chunk+1;
        b(i)=t;
    end

end
