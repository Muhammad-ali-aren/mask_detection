import torch
from tqdm.auto import tqdm
import numpy as np



def train_phase(model, train_loader,optimizer,loss_fun,device):
    train_accuracy = []
    train_loss = []
    model.train()
    for idx, (img, label) in enumerate(train_loader):
        yhat = model(img.to(device))
        loss = loss_fun(yhat.squeeze(),label.to(torch.float32))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        converted = (yhat > 0.5).int().squeeze()
        correct = (converted == label).sum().item()
        train_loss.append(loss.item())
        train_accuracy.append(100 * correct / yhat.size(0))
    return model,train_accuracy,train_loss
def test_phase(model,test_loader,loss_fun,device):
    test_accuracy = []
    test_loss = []
    model.eval()
    with torch.inference_mode():
        for idx,(img,label) in enumerate(test_loader):
            yhat = model(img.to(device))
            loss = loss_fun(yhat.squeeze(), label.to(torch.float32))
            test_loss.append(loss.item())
            converted = (yhat > 0.5).int().squeeze()
            correct = (converted == label).sum().item()
            test_accuracy.append(100 * correct / yhat.size(0))
    return test_accuracy,test_loss

def train_model(model,train_loader,test_loader,optimizer,loss_fun,epochs,device):
    model_results = {
        "train_accuracy" : [],
        "train_loss" : [],
        "test_accuracy": [],
        "test_loss": []
    }
    for epoch in tqdm(range(epochs)):
        model,train_accuracy,train_loss = train_phase(model,train_loader,optimizer,loss_fun,device)
        test_accuracy,test_loss = test_phase(model,test_loader,loss_fun,device)
        train_accuracy,train_loss = np.mean(train_accuracy),np.mean(train_loss)
        test_accuracy,test_loss = np.mean(test_accuracy),np.mean(test_loss)
        
        print(
            f"Epoch: {epoch + 1} | "
            f"Train Accuracy: {train_accuracy:.1f}% | "
            f"Train Loss: {train_loss:.4f} | "
            f"Test Accuracy: {test_accuracy:.1f}% | "
            f"Test Loss: {test_loss:.4f} | "
        )
        model_results['train_accuracy'].append(train_accuracy)
        model_results['train_loss'].append(train_loss)
        model_results['test_accuracy'].append(test_accuracy)
        model_results['test_loss'].append(test_loss)
    return model_results,model