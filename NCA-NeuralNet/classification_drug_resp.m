%% Pred data class
for i = 1:1:212
    for j = 1:1:43
        if(y(i,j) < thres(i,1))
            class_predicted(i,j) = 1
        else
            class_predicted(i,j) = 0
        end
    end
end
%% Orignal data classes
class_orig = []
for i = 1:1:212
    for j = 1:1:43
        if(t(i,j) < thres(i,1))
            class_orig(i,j) = 1
        else
            class_orig(i,j) = 0
        end
    end
end
%% Performance_wrt_drugs
a = 80
for a = 1:1:212
    cp = classperf(class_orig(a,:), class_predicted(a,:))
    accuracy{a} = cp.CorrectRate
    sensitivity{a} = cp.Sensitivity
    specificity{a} = cp.Specificity;
end
%% %% Performance_wrt_cell lines
a =11
for a = 1:1:43
    cp_cell = classperf(class_orig(:,a), class_predicted(:,a))
    accuracy_cell{a} = cp_cell.CorrectRate
    sensitivity_cell{a} = cp_cell.Sensitivity
    specificity_cell{a} = cp_cell.Specificity
end
    