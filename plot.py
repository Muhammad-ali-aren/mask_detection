import matplotlib.pyplot as plt

def plot_peformance_curves(results,EPOCHS):
    train_acc = results['train_accuracy']
    train_loss = results['train_loss']
    test_acc = results['test_accuracy']
    test_loss = results['test_loss']

    fig,ax = plt.subplots(1,2,figsize=(8,2))
    ax[0].plot(range(EPOCHS),train_acc,'o-',label="train acc")
    ax[0].plot(range(EPOCHS), test_acc,'-o',label='test acc')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('accuracy')
    ax[0].legend()
    ax[1].plot(range(EPOCHS), train_loss,'^-',label='train loss')
    ax[1].plot(range(EPOCHS), test_loss,'^-',label='test loss')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('loss')
    ax[1].legend()
    plt.show()
