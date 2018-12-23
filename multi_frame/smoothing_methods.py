
def smooth_by_majority(predictions, window=5):
    def majority(li, left, right):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        return 0 if sum(target) < (len(target)/2+1) else 1
        
    new_preds = [round(pred) for pred in predictions]
    wing = window/2
    for i in range(len(rpreds)):
        rpreds[i] = majority(rpreds, i-wing, i+wing)
    return rpreds

def pre_smooth_by_majority(predictions, window=5):
    def majority(li, left, right):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        return 0 if sum(target) < (len(target)/2+1) else 1
        
    new_preds = [round(pred) for pred in predictions]
    for i in range(len(new_preds)):
        rpreds[i] = majority(predictions, i-window, i)
    return rpreds

def smooth_predictions_by_average_score(predictions, window=5, threshold=0.5):
    def ave_score_over_threshold(li, left, right, threshold):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        ave = float(sum(target)) / len(target)
        return 0 if ave < threshold else 1
    
    new_preds = [0 for _ in range(len(predictions))]
    wing = window/2
    for i in range(len(predictions)):
         new_preds[i] = ave_score_over_threshold(predictions, i-wing, i+wing, threshold)
    return new_preds

def pre_smooth_predictions_by_average_score(predictions, window=5, threshold=0.5):
    def ave_score_over_threshold(li, left, right, threshold):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        ave = float(sum(target)) / len(target)
        return 0 if ave < threshold else 1
    
    new_preds = [0 for _ in range(len(predictions))]
    for i in range(len(predictions)):
         new_preds[i] = ave_score_over_threshold(predictions, i-window, i, threshold)
    return new_preds

def average_predictions(predictions, window=5):
    def average(li, left, right):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        ave = float(sum(target)) / len(target)
        return ave
    
    new_preds = [0 for _ in range(len(predictions))]
    wing = window/2
    for i in range(len(predictions)):
         new_preds[i] = average(predictions, i-wing, i+wing)
    return new_preds

def pre_average_predictions(predictions, window=5):
    def average(li, left, right):
        left = 0 if left<0 else left
        right = len(li)-1 if right>=len(li) else right
        target = li[left:right+1]
        ave = float(sum(target)) / len(target)
        return ave
    
    new_preds = [0 for _ in range(len(predictions))]
    for i in range(len(predictions)):
         new_preds[i] = average(predictions, i-window, i)
    return new_preds

def round_predictions(predictions, threshold=0.5):
    return [int(prediction>=threshold) for prediction in predictions]

