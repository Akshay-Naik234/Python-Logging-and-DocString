from utils.all_utils import prepare_data,save_plot
import pandas as pd
from utils.model import Perceptron

def main(data,modelName,plotName,eta,epoches):
    df_AND = pd.DataFrame(data)
    X,y = prepare_data(df_AND)
    model_and=Perceptron(eta=eta,epoches=epoches)
    model_and.fit(X,y)
    _ = model_and.total_loss()
    df_AND = pd.DataFrame(AND)
    model_and.save(filename=modelName)
    save_plot(df_AND,model_and,filename=plotName)


if __name__ == '__main__':
    AND = {
    "x1": [0,0,1,1],
    "x2": [0,1,0,1],
    "y" : [0,0,0,1]
    }
    ETA=0.1
    EPOCHES=10
    main(data=AND,modelName="and.model",plotName='and.png',eta=ETA,epoches=EPOCHES)






