
figure
plot(meas.Time,meas.Voltage,'r')
grid on
title ('Battery Voltage')
xlabel('Time (Sec)')
ylabel('Voltage (V)')
figure
plot(meas.Time,meas.Ah,'r')
grid on
title ('Battery Amp-Hours')
xlabel('Time (Sec)')
ylabel('Amp-Hours(Ah)')
figure
plot(meas.Time,meas.Current,'r')
hold on
grid on
title ('Battery Current')
xlabel('Time (Sec)')
ylabel('Current(A)')
figure
plot(meas.Time,meas.Power,'r')
hold on
grid on
title ('Battery Power')
xlabel('Time (Sec)')
ylabel('Power(W)')
figure
plot(meas.Time,meas.Battery_Temp_degC,'r')
hold on
grid on
title ('Battery Temprature')
xlabel('Time (Sec)')
ylabel('T(degC)')