import matplotlib.pyplot as plt
import numpy as np 
import cv2
import torch 


# 텍스트 저장 
def save_text(text, file_path):
    with open(file_path, 'a+') as file:  
        file.write(text + "\n") 

# 학습 도중 auc 그래프 저장
def save_auc_graph_train(iters, scores, file_path):
    plt.clf()
    plt.plot(iters, scores, c='royalblue') # auc

    # check best score
    scores_np = np.array(scores)
    best_idx = np.argmax(scores_np)
    best_itr = iters[best_idx]
    best_score = scores[best_idx]
    plt.scatter([best_itr],[best_score],c='darkorange',s=25, edgecolors='royalblue')
    plt.text(best_itr, best_score, f'{best_itr}: {best_score:.3f}', ha='left', va='bottom')

    plt.xlabel('Iteration')
    plt.ylabel('AUC')
    plt.savefig(file_path)

# 히트맵 저장
def save_heatmap(F_frame, target_frame, file_path):
    diff_map = torch.sum(torch.abs(F_frame - target_frame).squeeze(), 0)
    
    # Normalize to 0 ~ 255.
    diff_map -= diff_map.min()
    diff_map /= diff_map.max()
    diff_map *= 255

    diff_map = diff_map.cpu().detach().numpy().astype('uint8')
    heat_map = cv2.applyColorMap(diff_map, cv2.COLORMAP_JET)
    
    # save
    cv2.imwrite(file_path, heat_map)

# 이상 점수 저장
def save_score_graph(answers_idx, scores, file_path, x='Frame', y='Anomaly Score'):
    length = len(scores)
    plt.clf()
    plt.plot([num for num in range(length)],[score for score in scores]) # plotting
    plt.bar(answers_idx, max(scores), width=1, color='r',alpha=0.5) # check answer
    plt.xlabel(x)
    plt.ylabel(y)
    plt.savefig(file_path)

# ROC-CURVE 저장 
def save_roc_curve(fpr, tpr, auc, file_path):
    plt.clf()
    plt.plot(fpr, tpr, label=f'ROC curve (area = {auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel('False Positive Rate (FRR)')
    plt.ylabel('True Positive Rate (TRR)')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.savefig(file_path)