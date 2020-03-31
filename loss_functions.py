import tensorflow.keras.backend as K

''' OPTIONAL ADDITIONAL LOSS FUNCTIONS '''
# extra weight to pixel values > 0
def weightedLoss(yTrue,yPred):
    return K.sum((5**yTrue)*(K.abs(yTrue-yPred)))
  
# explicitly punish masks
def totalLoss(yTrue,yPred):
    return K.sum((K.abs(yTrue-yPred))*(K.abs(1-K.std(yPred))))

# MSE, but weight some pixel values higher
def weighted_mse(yTrue,yPred):
    return K.mean(yTrue*(yTrue-yPred)**2)

# try to include larger scale info
def largeScale_loss(yTrue,yPred):
    pixel_loss = K.mean((yTrue-yPred)**2)
    region_loss = 0
    return pixel_loss + region_loss

# include particle sums as constraint in weighted MSE
def constrained_weighted_mse(yTrue,yPred):
    mse_weight = 1
    constraint_weight = 1/K.mean(yTrue)
    mse = K.mean(mse_weight*K.square(yPred - yTrue))
    constraint = K.abs(K.sum(yTrue)-K.sum(yPred))
    return (constraint_weight*constraint)+mse

def constrained_weighted_mae(yTrue,yPred):
    mae_weight = 1
    constraint_weight = 1/K.mean(yTrue)
    mae = K.mean(mae_weight*K.abs((yPred - yTrue)))
    constraint = K.abs(K.sum(yTrue)-K.sum(yPred))
    return (constraint_weight*constraint)+mae

def constraint_component(yTrue,yPred):
    constraint = K.abs(K.sum(yTrue)-K.sum(yPred))
    constraint_weight = 1#/(48*48*48)
    return constraint_weight*constraint
