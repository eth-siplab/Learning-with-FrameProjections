import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

import numpy as np
import os
import pickle as cp
from data_preprocess.augmentations import gen_aug, DataTransform_TD, DataTransform_FD, data_transform_masked4cl
from new_augmentations import *
from models.frameworks import *
from models.loss import *
from models.backbones import *
from models.models_nc import ResNet1D, UNET_2D_simp, FourierEncoder, FourierAutoencoder, ConvMapping
from models.TFC.model_tfc import TFC
from models.TFC.tfc_loss import TFC_Loss
from sklearn.metrics import roc_auc_score
from data_preprocess import data_preprocess_IEEE_small
# from data_preprocess import data_preprocess_IEEE_big
# from data_preprocess import data_preprocess_dalia
from data_preprocess import data_preprocess_ucihar
from data_preprocess import data_preprocess_usc
from data_preprocess import data_preprocess_hhar
from data_preprocess import data_preprocess_clemson
from data_preprocess import data_preprocess_ecg
from data_preprocess import data_preprocess_sleep
from utils import TSNEPlotter

from sklearn.metrics import f1_score, accuracy_score
from scipy.special import softmax
import seaborn as sns
from copy import deepcopy

# create directory for saving models and plots
global model_dir_name
model_dir_name = 'results'
if not os.path.exists(model_dir_name):
    os.makedirs(model_dir_name)
global plot_dir_name
plot_dir_name = 'plot'
if not os.path.exists(plot_dir_name):
    os.makedirs(plot_dir_name)

def setup_dataloaders(args):
    if args.dataset == 'ieee_small':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_IEEE_small.prep_ieee_small(args)
    if args.dataset == 'ieee_big':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_IEEE_big.prep_ieeebig(args)     
    if args.dataset == 'dalia':
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 180 # 30 -- 210 bpm
        train_loaders, val_loader, test_loader = data_preprocess_dalia.prep_dalia(args)                       
    if args.dataset == 'clemson':
        args.n_feature = 1
        args.len_sw = 480
        args.n_class = 49
        train_loaders, val_loader, test_loader = data_preprocess_clemson.prep_clemson(args)   
    if args.dataset == 'ucihar':
        args.n_feature = 9
        args.len_sw = 128
        args.n_class = 6
        train_loaders, val_loader, test_loader = data_preprocess_ucihar.prep_ucihar(args)
    if args.dataset == 'usc':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 12
        train_loaders, val_loader, test_loader = data_preprocess_usc.prep_usc(args)
    if args.dataset == 'chapman':
        args.n_feature = 4
        args.len_sw = 1000
        n_class = 4 
        setattr(args, 'n_class', n_class)
        train_loaders, val_loader, test_loader = data_preprocess_ecg.prep_ecg(args)    
    if args.dataset == 'cpsc':
        args.n_feature = 4
        args.len_sw = 1000
        n_class = 9
        setattr(args, 'n_class', n_class)
        train_loaders, val_loader, test_loader = data_preprocess_ecg.prep_ecg(args)                  
    if args.dataset == 'hhar':
        args.n_feature = 6
        args.len_sw = 100
        args.n_class = 6
        source_domain = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i'] if args.cases == 'subject_large_ssl_fn' else ['a', 'b', 'c', 'd']
        # source_domain.remove(args.target_domain)
        train_loaders, val_loader, test_loader = data_preprocess_hhar.prep_hhar(args, SLIDING_WINDOW_LEN=args.len_sw, SLIDING_WINDOW_STEP=int(args.len_sw * 0.5),
                                                                                device=args.device,
                                                                                train_user=source_domain,
                                                                                test_user=args.target_domain)    
    if args.dataset == 'sleep': 
        args.n_feature = 1
        args.len_sw = 200
        args.n_class = 5
        train_loaders, val_loader, test_loader = data_preprocess_sleep.prep_sleep(args)    
    return train_loaders, val_loader, test_loader

def setup_linclf(args, DEVICE, bb_dim):
    '''
    @param bb_dim: output dimension of the backbone network
    @return: a linear classifier
    '''
    classifier = Classifier(bb_dim=bb_dim, n_classes=args.n_class)
    classifier.classifier.weight.data.normal_(mean=0.0, std=0.01)
    classifier.classifier.bias.data.zero_()
    classifier = classifier.to(DEVICE)
    return classifier

def setup_model_optm(args, DEVICE, classifier=True):
    # set up backbone network
    if args.backbone == 'FCN':
        backbone = FCN(in_dim=args.len_sw, n_channels=args.n_feature, n_classes=args.n_class, backbone=True)
    elif args.backbone == 'FCN2':
        backbone = FCN_2(in_dim= args.out_dim, n_channels=args.n_feature, n_classes=args.n_class, backbone=True)        
    elif args.backbone == 'DCL':
        backbone = DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=5, LSTM_units=args.out_dim, backbone=True)
    elif args.backbone == 'DCL2':
        backbone = DeepConvLSTM_2(n_channels=args.n_feature, n_classes=args.n_class, conv_kernels=64, kernel_size=7, LSTM_units=128, backbone=True)
    elif args.backbone == 'LSTM':
        backbone = LSTM(n_channels=args.n_feature, n_classes=args.n_class, LSTM_units=128, backbone=True)
    elif args.backbone == 'AE':
        backbone = AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, outdim=128, backbone=True)
    elif args.backbone == 'CNN_AE':
        backbone = CNN_AE(n_channels=args.n_feature, n_classes=args.n_class, out_channels=128, backbone=True)
    elif args.backbone == 'Transformer':
        backbone = Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class, dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=True)
    elif args.backbone == 'unet':
        backbone = UNET_1D_simp_ssl(input_dim=args.n_feature, output_dim=args.out_dim, layer_n=32, kernel_size=5, depth=1, args=args, backbone=True)
    elif args.backbone == 'resnet':
        backbone = ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5, stride=2, groups=1, n_block=args.block, n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, output_dim=args.out_dim, backbone=True)
    elif args.backbone == 'TFC_backbone':
        backbone = TFC_backbone(args, output_dim=2*args.out_dim)        
    else:
        NotImplementedError

    # set up model and optimizers
    if args.framework in ['byol', 'simsiam']:
        model = BYOL(DEVICE, backbone, window_size=args.len_sw, n_channels=args.n_feature, projection_size=args.p,
                     projection_hidden_size=args.phid, moving_average=args.EMA)
        optimizer1 = torch.optim.Adam(model.online_encoder.parameters(),
                                      args.lr,
                                      weight_decay=args.weight_decay)
        optimizer2 = torch.optim.Adam(model.online_predictor.parameters(),
                                      args.lr * args.lr_mul,
                                      weight_decay=args.weight_decay)
        optimizers = [optimizer1, optimizer2]
    elif args.framework == 'simclr' or args.framework == 'vicreg' or args.framework == 'barlowtwins': # Same models, different losses
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'nnclr':
        model = NNCLR(backbone=backbone, dim=args.p, pred_dim=args.phid)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'tstcc':
        model = TSTCC(backbone=backbone, DEVICE=DEVICE, temp_unit=args.temp_unit, tc_hidden=100)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]
    elif args.framework == 'ts2vec': # dummy models for ts2vec
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]
    elif args.framework == 'tfc':
        model = TFC(backbone=backbone, args=args, dim=2*args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=args.weight_decay)
        optimizers = [optimizer]     
    elif args.framework == 'clip':
        FT_backbone = FourierEncoder(in_channels=args.n_feature, in_length=args.len_sw, out_channels=args.n_feature, kernel_size=3)
        model = CLIP(backbone=backbone, backbone_FT=FT_backbone, DEVICE=DEVICE, dim=args.p, args=args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]  
    elif args.framework == 'mtm':
        model = MTM(backbone=backbone, DEVICE=DEVICE, dim=args.p, args=args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]                 
    elif args.framework == 'isoalign':
        spect_encoder = UNET_2D_simp(input_channels=args.n_feature, spect_freq=args.spect_freq, spect_time=args.spect_time, output_channels=args.n_feature, kernel_size=5, layer_n=32)
        FT_backbone = FourierEncoder(in_channels=args.n_feature, in_length=args.len_sw, out_channels=args.n_feature, kernel_size=3)
        FT_encoder = FourierAutoencoder(encoder=FT_backbone, input_channels=args.n_feature, in_length=args.len_sw)
        model = IsoAlign(backbone=backbone, spect_encoder=spect_encoder, FT_encoder=FT_encoder, DEVICE=DEVICE, dim=args.p, batch_size=args.batch_size, args=args)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        optimizers = [optimizer]
    else:
        NotImplementedError

    model = model.to(DEVICE)

    # set up linear classfier
    if classifier:
      # bb_dim = backbone.out_dim if args.framework not in ['isoalign'] else (backbone.out_dim) * 3
        if args.framework == 'isoalign':
            bb_dim = (backbone.out_dim) * 2 if (args.wo_OB or args.wo_OF) else (backbone.out_dim) * 3
            # bb_dim = (backbone.out_dim) * 2 if (args.wo_OB or args.wo_OF) else (backbone.out_dim) * 1 # second ablation
        else:
            bb_dim = backbone.out_dim
        classifier = setup_linclf(args, DEVICE, bb_dim)
        return model, classifier, optimizers

    else:
        return model, optimizers

def delete_files(args):
    model_dir = model_dir_name + '/pretrain_' + args.model_name + '.pt'
    if os.path.isfile(model_dir):
        os.remove(model_dir)

    cls_dir = model_dir_name + '/lincls_' + args.model_name + '.pt'
    if os.path.isfile(cls_dir):
        os.remove(cls_dir)

def setup(args, DEVICE):
    # set up default hyper-parameters
    if args.framework == 'byol':
        args.weight_decay = 1.5e-6
    if args.framework == 'simsiam':
        args.weight_decay = 1e-4
        args.EMA = 0.0
        args.lr_mul = 1.0
    if args.framework in ['simclr', 'nnclr']:
        args.criterion = 'NTXent'
        args.weight_decay = 1e-6
    if args.framework == 'tstcc':
        args.criterion = 'NTXent'
        args.backbone = 'FCN'
        args.weight_decay = 3e-4
    if args.framework == 'vicreg':
        args.criterion = 'VICReg'
        args.weight_decay = 1e-6
    if args.framework == 'barlowtwins':
        args.criterion = 'barlowtwins'
        args.weight_decay = 1.5e-6
    if args.framework == 'clip':
        args.criterion = 'clip'
        args.weight_decay = 1e-6      
    if args.framework == 'mtm':
        args.criterion = 'mtm'          
    if args.framework == 'tfc':
        args.criterion = 'tfc'
        args.backbone = 'TFC_backbone'        
    if args.framework == 'isoalign':
        args.criteria = 'isoalign'
        args.weight_decay = 1e-6

    model, classifier, optimizers = setup_model_optm(args, DEVICE, classifier=True)

    # loss fn
    if args.criterion == 'cos_sim':
        criterion = nn.CosineSimilarity(dim=1)
    elif args.criterion == 'NTXent':
        if args.framework == 'tstcc':
            criterion = NTXentLoss(args.batch_size, temperature=0.2, world_size=args.num_gpus)
        else:
            criterion = NTXentLoss(args.batch_size, temperature=0.1, world_size=args.num_gpus)
    elif args.criterion == 'Cont_InfoNCE':
        criterion = Cont_InfoNCE(DEVICE, args.batch_size, temperature=0.1)
    elif args.criterion == 'VICReg':
        criterion = VICReg(args)
    elif args.criterion == 'barlowtwins':
        criterion = BarlowTwins(args)
    elif args.criterion == 'clip':
        criterion = CLIP_loss(DEVICE, temperature=0.1)    
    elif args.criterion == 'mtm':
        criterion = MTM_loss(DEVICE, args)                    
    elif args.criterion == 'tfc':
        criterion = TFC_Loss(args)    

    args.model_name = 'try_scheduler_' + args.framework + '_pretrain_' + args.dataset + '_eps' + str(args.n_epoch) + '_lr' + str(args.lr) + '_bs' + str(args.batch_size) \
                      + '_aug1' + args.aug1 + '_aug2' + args.aug2 + '_dim-pdim' + str(args.p) + '-' + str(args.phid) \
                      + '_EMA' + str(args.EMA) + '_criterion_' + args.criterion + '_lambda1_' + str(args.lambda1) + '_lambda2_' + str(args.lambda2) + '_tempunit_' + args.temp_unit

    criterion_cls = nn.CrossEntropyLoss() 
    optimizer_cls = torch.optim.Adam(classifier.parameters(), lr=args.lr_cls)

    schedulers = []
    for optimizer in optimizers:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)
        schedulers.append(scheduler)

    global nn_replacer
    nn_replacer = None
    if args.framework == 'nnclr':
        nn_replacer = NNMemoryBankModule(size=args.mmb_size)

    global recon
    recon = None
    if args.backbone in ['AE', 'CNN_AE']:
        recon = nn.MSELoss()

    return model, optimizers, schedulers, criterion, classifier, criterion_cls, optimizer_cls

def calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=None, nn_replacer=None, view_learner=None):
    sample = sample.transpose(2,1) # sample --> (Batch_size, time steps, channel size)
    sample = sample.detach().cpu().numpy()

    aug_sample1, aug_sample2 = gen_aug(sample, args.aug1, args), gen_aug(sample, args.aug2, args) # Shape --> (Batch_size, number of inputs, channel size)

    aug_sample1, aug_sample2, target = aug_sample1.to(DEVICE).float(), aug_sample2.to(DEVICE).float(), target.to(DEVICE).long()

    if args.framework in ['byol', 'simsiam']:
        assert args.criterion == 'cos_sim'
    if args.framework in ['tstcc', 'simclr', 'nnclr']:
        assert args.criterion == 'NTXent'
    if args.framework in ['byol', 'simsiam', 'nnclr']:
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
        else:
            p1, p2, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
        if args.framework == 'nnclr':
            z1 = nn_replacer(z1, update=False)
            z2 = nn_replacer(z2, update=True)
        if args.criterion == 'cos_sim':
            loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
        elif args.criterion == 'NTXent':
            loss = (criterion(p1, z2) + criterion(p2, z1)) * 0.5
        if args.backbone in ['AE', 'CNN_AE']:
            loss = loss * args.lambda1 + recon_loss * args.lambda2
    if args.framework == 'simclr' or args.framework == 'vicreg' or args.framework == 'barlowtwins':
        if args.backbone in ['AE', 'CNN_AE']:
            x1_encoded, x2_encoded, z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            recon_loss = recon(aug_sample1, x1_encoded) + recon(aug_sample2, x2_encoded)
            loss = loss * args.lambda1 + recon_loss * args.lambda2
        else:
            z1, z2 = model(x1=aug_sample1, x2=aug_sample2)
            if args.aug3:
                aug_sample3 = gen_aug(sample, 'random', args).to(DEVICE).float()
                _, z3 = model(x1=aug_sample1, x2=aug_sample3)
                loss = criterion(z1, z3) + criterion(z1, z2) + criterion(z2, z3)
            else:
                loss = criterion(z1, z2)
    if args.framework == 'tstcc':
        nce1, nce2, p1, p2 = model(x1=aug_sample1, x2=aug_sample2)
        tmp_loss = nce1 + nce2
        ctx_loss = criterion(p1, p2)
        loss = tmp_loss * args.lambda1 + ctx_loss * args.lambda2
    if args.framework == 'ts2vec': # dummy models for ts2vec
        model = SimCLR(backbone=backbone, dim=args.p)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        optimizers = [optimizer]        
    if args.framework == 'tfc': # For tfc --aug 1 and --aug 2 should be 'na', it applies the augmentation from its bank
        aug_sample1_f = torch.abs(torch.fft.fft(aug_sample1, dim=1, norm='ortho'))

        aug_sample2 = DataTransform_TD(aug_sample1) # aug_sample1 is no aug.
        aug_sample2_f = DataTransform_FD(aug_sample1_f) 

        h_time, z_time, h_freq, z_freq = model(x_in_t=aug_sample1, x_in_f=aug_sample1_f)
        h_time_aug, z_time_aug, h_freq_aug, z_freq_aug = model(x_in_t=aug_sample2, x_in_f=aug_sample2_f)
        loss = criterion(z_time, z_time_aug, z_freq, z_freq_aug, h_time, h_time_aug, h_freq, h_freq_aug)  
    if args.framework == 'mtm':       
        data_masked_m, mask = data_transform_masked4cl(aug_sample1, args.masking_ratio, args.lm, args.positive_nums)
        z, h, x = model(aug_sample1, data_masked_m)           
        loss = criterion(z, h, x)          
    if args.framework == 'clip':
        aug_sample2 = torch.fft.rfft(aug_sample1, dim=1, norm='ortho')
        z1, z2 = model(aug_sample1, aug_sample2)           
        loss = criterion(z1, z2)        
    return loss

def train(train_dataset, val_loader, model, DEVICE, optimizers, schedulers, criterion, args):
    # Wrap the model in DistributedDataParallel
    model.to(DEVICE)
    if 'LOCAL_RANK' in os.environ:
        model = DDP(model, device_ids=[args.rank], find_unused_parameters=True) 
        train_sampler = DistributedSampler(train_dataset)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, drop_last=True)
    else:
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    best_model = None
    min_train_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler(init_scale=2**13) if args.use_amp else None
    
    # If possible, set drop_last=True in your DataLoader to avoid batch size mismatches
    for epoch in range(args.n_epoch):
        total_loss, n_batches = 0.0, 0
        model.train()
        train_sampler.set_epoch(epoch)  if 'LOCAL_RANK' in os.environ else model
        for idx, train_x in enumerate(train_loader):
            # Assuming train_x is a tuple: (sample, target, ...). For self-supervised frameworks like "isoalign",
            # target might not be used and an extra input (e.g., a second view) is available.
            sample = train_x[0].to(DEVICE)
            # Use target if available (may not be used in self-supervised learning)
            target = train_x[1].to(DEVICE) if len(train_x) > 1 and train_x[1] is not None else None
            cwt = train_x[2].to(DEVICE) if (args.framework == 'isoalign' and len(train_x) > 2) else None
            FT = train_x[3].to(DEVICE) if (args.framework == 'isoalign' and len(train_x) > 3) else None
            
            for optimizer in optimizers:
                optimizer.zero_grad()
            
            n_batches += 1
            with torch.cuda.amp.autocast(enabled=args.use_amp):
                if args.framework == 'isoalign':
                    loss = model(sample, cwt, FT)
                else:
                    loss = calculate_model_loss(args, sample, target, model, criterion, DEVICE, recon=recon, nn_replacer=nn_replacer)

            # import pdb; pdb.set_trace()
            if args.use_amp:
                scaler.scale(loss).backward()
                for optimizer in optimizers:
                    scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                for optimizer in optimizers:
                    optimizer.step()

            if args.framework in ['byol', 'simsiam']:
                model.update_moving_average()

            if dist.is_available() and dist.is_initialized():
                loss = loss.data.clone()
                dist.all_reduce(loss.div_(dist.get_world_size()))

            total_loss += loss.item()

        # Update schedulers after each epoch
        for scheduler in schedulers:
            scheduler.step()
            
        avg_train_loss = total_loss / n_batches if n_batches > 0 else total_loss
        
        # Only rank 0 writes logs/checkpoints
        if not ('LOCAL_RANK' in os.environ) or dist.get_rank() == 0:
            if avg_train_loss < min_train_loss:
                min_train_loss = avg_train_loss
                best_model = copy.deepcopy(model.state_dict())  if not 'LOCAL_RANK' in os.environ else copy.deepcopy(model.module.state_dict())
                # Save the best model dictionary to a results folder.
                save_path = os.path.join('results', f"{args.model_name}_best.pt")
                torch.save({'model_state_dict': best_model}, save_path)
                # Save checkpoint if needed.
                # torch.save({'model_state_dict': best_model}, "checkpoint.pt")
            # print(f"Epoch {epoch}: Train Loss = {avg_train_loss:.4f}")

    if 'LOCAL_RANK' in os.environ:
        # Broadcast best_model from rank 0 to all processes
        best_model_list = [best_model]
        dist.broadcast_object_list(best_model_list, src=0)
        best_model = best_model_list[0]

    return best_model

def test_isoalign(data_loader, model, args): # Test the pre-trained model --> to observe the features
    train_loader = DataLoader(data_loader, batch_size=args.batch_size, shuffle=True, drop_last=False)
    tsne_plotter = TSNEPlotter(perplexity=30, n_iter=1000, uniform_color=True, random_state=None)

    time_embedded, cwt_embedded, FT_embedded = None, None, None
    labels = [] 
    with torch.no_grad():
        model.eval()
        
        for idx, testx in enumerate(train_loader):
            sample, target, cwt, FT = testx[0], testx[1], testx[2], testx[3]
            _, time_encoded = model.encoder(sample.transpose(2,1))
            _, cwt_encoded = model.spect_encoder(cwt.permute(0, 3, 1, 2))
            _, FT_encoded = model.FT_encoder(FT)

            time_embedded = time_encoded if time_embedded is None else torch.cat((time_embedded, time_encoded), 0)
            cwt_embedded = cwt_encoded if cwt_embedded is None else torch.cat((cwt_embedded, cwt_encoded), 0)
            FT_embedded = FT_encoded if FT_embedded is None else torch.cat((FT_embedded, FT_encoded), 0)
            labels.append(target)

    labels = torch.cat(labels, dim=0).detach().cpu()
    # tsne_plotter.plot(time_embedded.detach().cpu(), labels, title="TSNE Plot", save_path="plot/tsne_time_T.png")
    # tsne_plotter.plot(cwt_embedded.detach().cpu(), labels, title="TSNE Plot", save_path="plot/tsne_cwt_T.png")
    # tsne_plotter.plot(FT_embedded.detach().cpu(), labels, title="TSNE Plot", save_path="plot/tsne_FT_T.png")
    time_embedded = time_embedded[0:20000,:]
    cwt_embedded = cwt_embedded[0:20000,:]
    FT_embedded = FT_embedded[0:20000,:]    
    ################    
    corr_coeff, d1, d2 = tsne_plotter.plot_distance_scatter(time_embedded.detach().cpu(), cwt_embedded.detach().cpu(), cmap='crest', save_path="plot/distance_scatter_cwt_hhar.png")    
    angle_matrix = tsne_plotter.plot_misalignment_density(time_embedded.detach().cpu(), FT_embedded.detach().cpu(), bins=5, palette='rocket', save_path="plot/misalign_FT_pair.png")
    angle_matrix_all = tsne_plotter.plot_all_misalignment_density(time_embedded.detach().cpu(), FT_embedded.detach().cpu(), bins=5, palette='rocket', save_path="plot/misalign_FT_all.png")
    # angles = tsne_plotter.plot_misalignment(time_embedded.detach().cpu(), FT_embedded.detach().cpu(), save_path="plot/misalign_spec.png")
    # angle_matrix = tsne_plotter.plot_all_misalignment(time_embedded.detach().cpu(), FT_embedded.detach().cpu(), save_path="plot/misalign_all.png")
    return model

def test(data_loader, best_model, DEVICE, criterion, args): # Test the pre-trained model --> to observe the features if desired
    model, _ = setup_model_optm(args, DEVICE, classifier=False)
    model.load_state_dict(best_model)

    # if args.framework in ['isoalign']:
    #     test_isoalign(data_loader, model, args)
    return model

def lock_backbone(model, args):
    if args.framework not in ['ts2vec']:
        for name, param in model.named_parameters():
            param.requires_grad = False
    else:
        for name, param in model._net.named_parameters():
            param.requires_grad = False

    if args.framework in ['simsiam', 'byol']:
        trained_backbone = model.online_encoder.net
    elif args.framework in ['simclr', 'simper', 'nnclr', 'tstcc','vicreg', 'barlowtwins', 'clip', 'mtm']:
        trained_backbone = model.encoder
    elif args.framework in ['ts2vec', 'isoalign', 'tfc']:
        trained_backbone = model
    else:
        NotImplementedError

    return trained_backbone

def calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args):
    sample = sample.transpose(2,1) # sample --> (Batch_size, time steps, channel size)
    target = target.round().long()

    if args.framework not in ['ts2vec', 'tfc']:
        _, feat = trained_backbone(sample)
    elif args.framework in ['tfc']:
        sample_f = torch.abs(torch.fft.fft(sample, dim=1, norm='ortho'))
        h_t, z_t, h_f, z_f = trained_backbone(sample, sample_f)
        feat = torch.cat((z_t, z_f), dim=1)        
    else:
        feat = torch.from_numpy(trained_backbone.encode(sample.detach().cpu().numpy(), encoding_window='full_series')).to(args.cuda)

    if len(feat.shape) == 3:
        feat = feat.reshape(feat.shape[0], -1)

    output = classifier(feat).squeeze()

    loss = criterion(output, target)
    _, predicted = torch.max(output.data, 1)
    return loss, predicted, feat, output

def train_lincls(train_loaders, val_loader, trained_backbone, classifier, DEVICE, optimizer, criterion, args):
    best_lincls = None
    min_val_loss = 1e8

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epoch, eta_min=0)

    for epoch in range(args.n_epoch):
        classifier.train()
        for idx, train_x in enumerate(train_loaders):
            sample, target = train_x[0], train_x[1]

            loss, predicted, _, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # save model
        # model_dir = model_dir_name + '/lincls_' + args.model_name + str(epoch) + '.pt'
        # if args.framework not in ['ts2vec']:
            # torch.save({'trained_backbone': trained_backbone.state_dict(), 'classifier': classifier.state_dict()}, model_dir) 

        if args.scheduler:
            scheduler.step()

        if args.cases in ['subject', 'subject_large','subject_large_ssl_fn']:
            with torch.no_grad():
                best_lincls = copy.deepcopy(classifier.state_dict())
        else:
            with torch.no_grad():
                classifier.eval()
                total_loss = 0
                total = 0
                correct = 0
                for idx, (sample, target, domain) in enumerate(val_loader):
                    sample, target = sample.to(DEVICE).float(), target.to(DEVICE)
                    loss, predicted, _, _ = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
                    total_loss += loss.item()
                    total += target.size(0)
                    correct += (predicted == target).sum()
                acc_val = float(correct) * 100.0 / total
                if total_loss <= min_val_loss:
                    min_val_loss = total_loss
                    best_lincls = copy.deepcopy(classifier.state_dict())
    return best_lincls

def test_lincls(test_loader, trained_backbone, best_lincls, DEVICE, criterion, args):  # Test the fine-tuned model
    if args.framework == 'ts2vec':
        classifier = setup_linclf(args, DEVICE, args.p)
    else:
        classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim)
    # classifier = setup_linclf(args, DEVICE, trained_backbone.out_dim) if args.framework not in ['ts2vec'] else setup_linclf(args, DEVICE, args.p)
    classifier.load_state_dict(best_lincls)

    total_loss = 0
    all_feats, all_trgs, all_prds, all_outputs = [], [], [], []

    with torch.no_grad():
        classifier.eval()
        
        for idx, testx in enumerate(test_loader):
            sample, target = testx[0], testx[1]
            
            loss, predicted, feat, outputs = calculate_lincls_output(sample, target, trained_backbone, classifier, criterion, args)
            
            total_loss += loss.item()

            all_feats.append(feat)
            all_trgs.append(target.data.cpu().numpy())
            all_prds.append(predicted.data.cpu().numpy())
            all_outputs.append(outputs.data.cpu().numpy())

    feats = torch.cat(all_feats, 0)  
    trgs = np.concatenate(all_trgs, axis=0)  
    prds = np.concatenate(all_prds, axis=0)
    outputs = np.concatenate(all_outputs, axis=0)  

    if args.dataset in ['ieee_small', 'ieee_big', 'dalia']:
        m1 = np.mean(np.abs(trgs - prds)) # MAE
        m2 = np.sqrt(np.mean((trgs - prds) ** 2))  # RMSE
        m3 = np.corrcoef(trgs, prds)[0,1] # correlation
        if np.isnan(m3): 
            m3 = 0
    elif args.dataset in ['ucihar', 'usc', 'hhar']: # ACC | F1 | W-F1
        m1 = accuracy_score(trgs, prds) * 100
        m2 = f1_score(trgs, prds, average='macro') * 100
        m3 = f1_score(trgs, prds, average='weighted') * 100
    elif args.dataset == 'clemson':
        trgs, prds = trgs + 20, prds + 20
        m1 = 100*np.mean(np.abs((trgs-prds)/trgs))
        m2 = np.mean(np.abs(trgs-prds))
        m3 = np.sqrt(np.mean((trgs - prds) ** 2))  # RMSE
    elif args.dataset in ['chapman', 'cpsc']: # ACC | F1 | AUC
        m1 = accuracy_score(trgs, prds) * 100
        m2 = f1_score(trgs, prds, average='macro') * 100
        otp1 =  softmax(outputs, axis=1)
        m3 = roc_auc_score(trgs, otp1, multi_class='ovo') * 100
    else:
        raise ValueError('Not a defined dataset')
        
    if args.plt == True:
        tsne(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_tsne.png')
        mds(feats, trgs, save_dir=plot_dir_name + '/' + args.model_name + '_mds.png')
        print('plots saved to ', plot_dir_name)
    return np.array([m1, m2, m3])

def train_mappers(train_loaders, model, DEVICE, args):

    for name, param in model.named_parameters():
            param.requires_grad = False

    map_ft = ConvMapping(in_channels=128, hidden_channels=1, kernel_size=3).to(DEVICE)
    map_cwt = ConvMapping(in_channels=128, hidden_channels=1, kernel_size=3).to(DEVICE)
    # Ablation 
    # map_ft = nn.Linear(128, 128).to(DEVICE)
    # map_cwt = nn.Linear(128, 128).to(DEVICE)
    # Ablation
    # map_ft = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)).to(DEVICE)
    # map_cwt = nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 128)).to(DEVICE)

    optimizer_ft = torch.optim.Adam(map_ft.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    optimizer_cwt = torch.optim.Adam(map_cwt.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    train_loader = DataLoader(train_loaders, batch_size=args.batch_size, shuffle=True, drop_last=True)

    best_model = None
    min_train_loss = float('inf')
    
    scaler = torch.cuda.amp.GradScaler(init_scale=2**13) if args.use_amp else None
    
    for epoch in range(args.n_epoch):
        total_loss, n_batches = 0.0, 0
        map_ft.train()
        map_cwt.train()
        model.eval()  # The original model remains frozen.
            
        for idx, train_x in enumerate(train_loader):
            sample = train_x[0].to(DEVICE)  
            cwt    = train_x[2].to(DEVICE)    
            FT     = train_x[3].to(DEVICE)    

            # Zero gradients for both optimizers.
            optimizer_ft.zero_grad()
            optimizer_cwt.zero_grad()
            n_batches += 1

            with torch.cuda.amp.autocast(enabled=args.use_amp):
                _, time_emb = model.encoder(sample.transpose(2,1))
                _, cwt_emb  = model.spect_encoder(cwt.permute(0,3,1,2))
                _, FT_emb   = model.FT_encoder(FT)
                
                ft_mapped  = map_ft(time_emb)
                cwt_mapped = map_cwt(time_emb)
                
                # Compute separate MSE losses.
                loss_ft = torch.nn.L1Loss()(ft_mapped, time_emb)
                loss_cwt = torch.nn.L1Loss()(cwt_mapped, cwt_emb)

                # loss_ft  = F.mse_loss(ft_mapped, FT_emb) # Ablation
                # loss_cwt = F.mse_loss(cwt_mapped, cwt_emb) # Ablation

            if args.use_amp:
                scaler.scale(loss_ft).backward()
                scaler.step(optimizer_ft)
                optimizer_ft.zero_grad()
                scaler.update()
            else:
                loss_ft.backward()
                optimizer_ft.step()

            if args.use_amp:
                scaler.scale(loss_cwt).backward()
                scaler.step(optimizer_cwt)
                optimizer_cwt.zero_grad()
                scaler.update()
            else:
                loss_cwt.backward()
                optimizer_cwt.step()

            total_loss += (loss_ft.item() + loss_cwt.item())

        avg_train_loss = total_loss / (n_batches * 2)  # average over both losses
        
        # Save the best mappers (i.e. best state of both mapping networks)
        if avg_train_loss < min_train_loss:
            min_train_loss = avg_train_loss
            best_model = {
                "map_ft": copy.deepcopy(map_ft.state_dict()),
                "map_cwt": copy.deepcopy(map_cwt.state_dict())
            }
            save_path = os.path.join('results', f"{args.model_name}_best.pt")
            torch.save(best_model, save_path)

    return_model = IntegratedEncoder(model, map_ft, map_cwt, args, out_dim=model.encoder.out_dim)

    return return_model


def build_model(args, part=None):
    """Instantiate and return a model based on args.backbone and related options."""

    if args.backbone == 'FCN':
        return FCN(n_channels=args.n_feature, n_classes=args.n_class, in_dim=args.len_sw, backbone=False)

    elif args.backbone == 'FCN_b':
        return FCN_big(n_channels=args.n_feature, n_classes=args.n_class, backbone=False, args=args)

    elif args.backbone == 'DCL':
        return DeepConvLSTM(n_channels=args.n_feature, n_classes=args.n_class,
                            conv_kernels=64, kernel_size=5, LSTM_units=128, backbone=False)

    elif args.backbone == 'LSTM':
        return LSTM(n_channels=args.n_feature, n_classes=args.n_class,
                    LSTM_units=128, backbone=False)

    elif args.backbone == 'AE':
        return AE(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                  outdim=128, backbone=False)

    elif args.backbone == 'CNN_AE':
        return CNN_AE(n_channels=args.n_feature, n_classes=args.n_class,
                      out_channels=128, backbone=False)

    elif args.backbone == 'Transformer':
        return Transformer(n_channels=args.n_feature, len_sw=args.len_sw, n_classes=args.n_class,
                           dim=128, depth=4, heads=4, mlp_dim=64, dropout=0.1, backbone=False)

    elif args.backbone == 'wavelet':
        return ScatterWave(args)

    elif args.backbone == 'WaveletNet':
        return WaveletNet(args=args)

    elif args.backbone == 'ModernTCN':
        return ModernTCN(args=args, class_num=args.n_class, seq_len=args.len_sw)

    elif args.backbone == 'resnet':
        return ResNet1D(in_channels=args.n_feature, base_filters=32, kernel_size=5,
                            stride=args.stride, groups=1, n_block=args.block,
                            n_classes=args.n_class, downsample_gap=2, increasefilter_gap=4, backbone=False)
    else:
        raise NotImplementedError(f"Backbone {args.backbone} not implemented.")