

t = [0:0.005:1];
x = sin(t*23) + 0.9*sin(t*52) + .7*sin(t*80) + .4*sin(t*120);

y = fft(x);

f = (0:length(y)-1)*100/length(y); 

subplot(2,1,1)
plot(f,abs(y))
title('Magnitude')

subplot(2,1,2)
plot(f,unwrap(angle(y))*180/pi)
title('Phase')