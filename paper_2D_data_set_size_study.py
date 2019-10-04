import torch
from matplotlib import pyplot as plt
from torch.utils import data
import models


torch.manual_seed(2)

n_in = 2
n_h1 = 100
n_h2 = 50
n_o = 1

n_o_uc = 2

# Using the same constant Neural network size for both models,
# compare performance as the measurement data size is increased

n_data_tests = [100, 200, 300, 400, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000]
# n_data_tests = [1000, 3000]
n_tests = len(n_data_tests)

n_trials = 2   # number of trials per each test

def vector_field(x, y, a=0.01):
    v1 = torch.exp(-a*x*y)*(a*x*torch.sin(x*y) - x*torch.cos(x*y))
    v2 = torch.exp(-a*x*y)*(y*torch.cos(x*y) - a*y*torch.sin(x*y))
    return (v1, v2)


# pregenerate validation data
x_val = 4.0*torch.rand(2000,2)
x1_val = x_val[:, 0].unsqueeze(1)
x2_val = x_val[:, 1].unsqueeze(1)

(v1, v2) = vector_field(x1_val, x2_val)
y1_val = v1 + 0.1 * torch.randn(x1_val.size())
y2_val = v2 + 0.1 * torch.randn(x1_val.size())
y_val = torch.cat((y1_val, y2_val), 1)


# Get the true function values on a grid
xv, yv = torch.meshgrid([torch.arange(0.0,20.0)*4.0/20.0, torch.arange(0.0,20.0)*4.0/20.0])
(v1,v2) = vector_field(xv, yv)


rms_error_save = torch.empty(n_tests, 1)
rms_error_save_uc = torch.empty(n_tests, 1)

print('running tests')
for test in range(n_tests):
    for trial in range(n_trials):
        # reinitialise models
        if 'model' in locals():
            del model
        if 'model_uc' in locals():
            del model_uc

        model = models.DerivNet2D(n_in, n_h1, n_h2, n_o)

        model_uc = torch.nn.Sequential(
            torch.nn.Linear(n_in, n_h1),
            torch.nn.Tanh(),
            torch.nn.Linear(n_h1, n_h2),
            torch.nn.Tanh(),
            torch.nn.Linear(n_h2, n_o_uc),
        )

        ## generate data and put into a data loader
        n_data = n_data_tests[test]

        x_train = 4.0*torch.rand(n_data,2)
        x1_train = x_train[:, 0].unsqueeze(1)
        x2_train = x_train[:, 1].unsqueeze(1)

        (v1_t, v2_t) = vector_field(x1_train, x2_train)
        y1_train = v1_t + 0.1 * torch.randn(x1_train.size())
        y2_train = v2_t + 0.1 * torch.randn(x1_train.size())

        # now put data in a convenient dataset and data loader


        training_set = models.Dataset(x1_train,x2_train,y1_train,y2_train)

        # data loader Parameters
        DL_params = {'batch_size': 100,
                  'shuffle': True,
                  'num_workers': 4}
        training_generator = data.DataLoader(training_set, **DL_params)

        train_iters = 300
        ## train constrained
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)


        loss_save = torch.empty(train_iters, 1)
        val_loss_save = torch.empty(train_iters, 1)
        min_val_loss = 1e10

        for epoch in range(train_iters):
            for x1_train, x2_train, y1_train, y2_train in training_generator:
                optimizer.zero_grad()
                x_train = torch.cat((x1_train, x2_train), 1)

                (yhat, v1hat, v2hat) = model(x_train)
                loss = (criterion(y1_train, v1hat) + criterion(y2_train, v2hat))/2 # divide by 2 as it is a mean
                loss.backward()
                optimizer.step()
            loss_save[epoch, 0] = loss



            (yhat, v1hat, v2hat) = model(x_val)
            val_loss = (criterion(y1_val, v1hat) + criterion(y2_val, v2hat))/2 # divide by 2 as it is a mean
            val_loss_save[epoch,0] = val_loss
            scheduler.step(epoch)
            # print('epoch: ', epoch, ' val loss: ', val_loss.item())


            if val_loss*1.01 < min_val_loss:
                min_val_loss = val_loss
                last_decrease = epoch
            else:
                if (epoch > 41 + last_decrease) and (epoch > 60):
                    break

        # work out the rms error for this one
        x_pred = torch.cat((xv.reshape(20 * 20, 1), yv.reshape(20 * 20, 1)), 1)
        (f_pred, v1_pred, v2_pred) = model(x_pred)
        error_new = torch.cat((v1.reshape(400,1) - v1_pred.detach(),v2.reshape(400,1) - v2_pred.detach()),0)
        rms_new = torch.sqrt(sum(error_new * error_new) / 800)
        rms_error_save[test] = rms_new/n_trials     # will be an average over multiple trials
        print('test: ', test, ' trial: ', trial, 'constrained rms error', rms_new.item())

        ## train unconstrained
        criterion = torch.nn.MSELoss()
        optimizer = torch.optim.Adam(model_uc.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 25, gamma=0.5, last_epoch=-1)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.25, last_epoch=-1)


        loss_save_uc = torch.empty(train_iters, 1)
        val_loss_save_uc = torch.empty(train_iters, 1)
        min_val_loss = 1e10


        for epoch_uc in range(train_iters):
            for x1_train, x2_train, y1_train, y2_train in training_generator:
                optimizer.zero_grad()
                x_train = torch.cat((x1_train, x2_train), 1)
                vhat = model_uc(x_train)
                y_train = torch.cat((y1_train, y2_train), 1)
                loss = criterion(y_train, vhat)
                loss.backward()
                optimizer.step()
            loss_save_uc[epoch_uc, 0] = loss


            (vhat) = model_uc(x_val)
            val_loss = criterion(y_val, vhat)
            val_loss_save_uc[epoch_uc,0] = val_loss
            scheduler.step(epoch_uc)
            # print('epoch: ', epoch_uc, ' val loss: ', val_loss.item())


            if val_loss*1.01 < min_val_loss:
                min_val_loss = val_loss
                last_decrease = epoch_uc
            else:
                if (epoch_uc > 41 + last_decrease) and (epoch_uc > 60):
                    break

        # work out the rms error for this trial
        (v_pred_uc) = model_uc(x_pred)
        v1_pred_uc = v_pred_uc[:, 0]
        v2_pred_uc = v_pred_uc[:, 1]

        error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(),v2.reshape(400) - v2_pred_uc.detach()),0)
        rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)

        rms_error_save_uc[test] = rms_uc/n_trials     # will be an average over multiple trials
        print('test: ', test, ' trial: ', trial, 'unconstrained rms error', rms_uc.item())





# plot the predicted function
x_pred = torch.cat((xv.reshape(20*20,1), yv.reshape(20*20,1)),1)
(f_pred, v1_pred, v2_pred) = model(x_pred)
(v_pred_uc) = model_uc(x_pred)
v1_pred_uc = v_pred_uc[:,0]
v2_pred_uc = v_pred_uc[:,1]

with torch.no_grad():
    # Initialize plot
    f,ax = plt.subplots(1, 1, figsize=(4, 3))
    ax.plot(n_data_tests,rms_error_save.detach().numpy())
    ax.plot(n_data_tests,rms_error_save_uc.detach().numpy(),color='r')
    ax.set_xlabel('Number of observations')
    ax.set_ylabel('rms error')
    ax.legend(['our approach','unconstrained'])
    plt.show()
    # f.savefig('sim_n_data_study.eps', format='eps')

    # # Initialize second plot
    # f2, ax2 = plt.subplots(1, 3, figsize=(13, 4))
    # Q = ax2[0].quiver(xv, yv, v1, v2, scale=None, scale_units='inches')
    # Q._init()
    # assert isinstance(Q.scale, float)
    # ax2[0].quiver(x1_train, x2_train, y1_train, y2_train, scale=Q.scale, scale_units='inches', color='r')
    # ax2[0].set_xlabel('$x_1$')
    # ax2[0].set_ylabel('$x_2$')
    #
    #
    # error_new = torch.cat((v1.reshape(400,1) - v1_pred.detach(),v2.reshape(400,1) - v2_pred.detach()),0)
    # rms_new = torch.sqrt(sum(error_new * error_new) / 800)
    #
    # ax2[1].quiver(xv, yv, v1-v1_pred.reshape(20,20).detach(), v2-v2_pred.reshape(20,20).detach(), scale=Q.scale, scale_units='inches')
    # ax2[1].set_xlabel('$x_1$')
    # ax2[1].set_ylabel('$x_2$')
    # ax2[1].set_title('Our Approach RMS error ={0:.2f}'.format(rms_new.item()))
    #
    # error_uc = torch.cat((v1.reshape(400) - v1_pred_uc.detach(),v2.reshape(400) - v2_pred_uc.detach()),0)
    # rms_uc = torch.sqrt(sum(error_uc * error_uc) / 800)
    #
    # ax2[2].quiver(xv, yv, v1-v1_pred_uc.reshape(20,20).detach(), v2-v2_pred_uc.reshape(20,20).detach(), scale=Q.scale, scale_units='inches')
    # ax2[2].set_xlabel('$x_1$')
    # ax2[2].set_ylabel('$x_2$')
    # ax2[2].set_title('Unconstrained NN RMS error ={0:.2f}'.format(rms_uc.item()))
    # # f2.savefig('div_free_fields.eps', format='eps')
    # plt.show()
