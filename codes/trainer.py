import torch
import logging
import numpy as np
import torch.nn.functional as F
import torch.optim as optim
from run_manager import RunManager
from torch.utils.data import DataLoader
import datetime
import requests
from early_stopping import EarlyStopping
import constants
from tqdm import tqdm
from sklearn.metrics import *
from models import FocalLoss
from torch.utils.tensorboard import SummaryWriter


def train(model, train_set, dev_set, test_set, hyper_params, batch_size, device, test, alpha=0.03, simi=False, local_rank=None):

    writer = SummaryWriter()

    train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=0, shuffle=True)
    m = RunManager()
    # embedder_params = list(map(id, model.embedder.parameters()))
    # params = filter(lambda p: id(p) not in embedder_params, model.parameters())
    # optimizer = optim.AdamW([{'params': params},
    #                          {'params': model.embedder.parameters(), 'lr': hyper_params.learning_rate_fine}],
    #                         lr=hyper_params.learning_rate)

    betas = (0.9, 0.999)
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                          lr=hyper_params.learning_rate, betas=betas, weight_decay=0)

    args = constants.get_args()
    dev_loader = DataLoader(dev_set, batch_size=batch_size, num_workers=0, shuffle=False)

    early_stopping = EarlyStopping(args.patience_1, verbose=True)
    
    # if test 
    if test:
        checkpoint_path = '../results/' + args.pre_file
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)['model'])

        print('loading model finished')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        
        model.eval()
    
    else:

        logging.info("Training Started...")
        m.begin_run(hyper_params, model, train_loader)

        epoch_start = 0
        print('total epoch: ', hyper_params.num_epoch)

        for epoch in range(epoch_start, hyper_params.num_epoch):
            m.begin_epoch(epoch + 1)
            model.train()
            bar = tqdm(train_loader)
            bar.set_description("Training")
            for batch in bar:
                #print('begin')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                texts = batch['text']
                lens = batch['length']
                targets = batch['codes']
                hadm_id = batch['hadm_id']
                
                lens, ids = torch.sort(lens, descending=True)

                texts = texts[ids].to(device)
                targets = targets[ids].to(device)

                #print('0')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                outputs, ldam_outputs, _ = model(texts, lens, targets)
                
                #print('1')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                if simi:
                    loss = F.binary_cross_entropy(outputs, targets)
                else:
                    if ldam_outputs is not None:
                        loss = F.binary_cross_entropy_with_logits(ldam_outputs, targets)
                    else:
                        loss = F.binary_cross_entropy_with_logits(outputs, targets)
                        #loss = FocalLoss(gamma=1)(outputs, targets)

                #print('2')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                optimizer.zero_grad()

                #print('2.1')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                loss.backward()

                #print('2.2')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                optimizer.step()

                #print('3')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

                m.track_loss(loss)
                # m.track_num_correct(preds, affinities)

                #print('4')
                #print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            writer.add_scalar('Loss/train', m.epoch_loss/len(train_loader.dataset), epoch)
            m.end_epoch()

            # Validation
            preds, probabs, targets, _, _, eval_loss = evaluate(model, dev_loader, device, dtset='dev', alpha=alpha, simi=simi)
            writer.add_scalar('Loss/validation', eval_loss, epoch)
            _, _, f1_score_micro = micro_f1(np.array(targets), preds)
            writer.add_scalar('f1_micro', f1_score_micro, epoch)
            checkpoint_path = early_stopping(f1_score_micro, model, epoch, args.name, probabs, local_rank)
            if early_stopping.early_stop:
                break

        m.end_run()

        hype = '_'.join([f'{k}_{v}' for k, v in hyper_params._asdict().items()])
        m.save(f'../results/check_{args.name}/train_results_{hype}')
        logging.info("Training finished.\n")


        # 加载最好的那次参数
        model.load_state_dict(torch.load(checkpoint_path)['model'])
        print('Best epoch: ', torch.load(checkpoint_path)['epoch'])

    # Training
    print("start loading training info")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=0)

    print("starting evaluating")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    preds, probabs, targets, _, _, _ = evaluate(model, train_loader, device, dtset='train',alpha = alpha, simi=simi )

    print("start computing the score")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    compute_scores(probabs, targets, hyper_params, dtset='train', simi=simi)

    print("finished")
    print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

    # Validation
    dev_loader = DataLoader(dev_set, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, probabs, targets, _, _, _ = evaluate(model, dev_loader, device, dtset='dev', alpha=alpha, simi=simi)
    compute_scores(probabs, targets, hyper_params, dtset='dev', simi=simi)
    
    time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    # test_dataset
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0)
    preds, probabs, targets, full_hadm_ids, full_attn_weights, _ = evaluate(model, test_loader, device, dtset='test', alpha=alpha, simi=simi)
    compute_scores(probabs, targets, hyper_params, dtset='test', full_hadm_ids=full_hadm_ids,
                   full_attn_weights=full_attn_weights, simi=simi)

    # index = np.where(np.array[full_hadm_ids]=='102295.0')
    # print(index)
    # print(np.where(np.array(preds[index]))==1)


def evaluate(model, loader, device, dtset, simi=False, alpha=0.02):
    if simi:
        fin_preds = []
        fin_targets = []
        fin_probabs = []
        full_hadm_ids = []
        full_attn_weights = []

        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            epoch_loss = 0
            for batch in loader:
                hadm_ids = batch['hadm_id']
                texts = batch['text']
                lens = batch['length']
                targets = batch['codes']

                texts = texts.to(device)
                targets = targets
                outputs, _, attn_weights = model(texts, lens)
                loss = F.binary_cross_entropy(outputs, targets.to(device) )
                epoch_loss += loss.item() * loader.batch_size

                fin_targets.extend(targets.tolist())
                fin_probabs.extend(outputs.detach().cpu().tolist())
                fin_preds.extend((outputs > alpha).int().detach().cpu().tolist())
              
            loss = epoch_loss / len(loader.dataset)
        return fin_preds, fin_probabs, fin_targets, full_hadm_ids, full_attn_weights, loss
    else:
        fin_targets = []
        fin_probabs = []
        full_hadm_ids = []
        full_attn_weights = []

        with torch.no_grad():
            # Set the model to evaluation mode
            model.eval()
            epoch_loss = 0
            for batch in tqdm(loader, unit="batches", desc="Evaluating"):
                hadm_ids = batch['hadm_id']
                texts = batch['text']
                lens = batch['length']
                targets = batch['codes']

                texts = texts.to(device)
                targets = targets
                outputs, _, attn_weights = model(texts, lens)

                loss = F.binary_cross_entropy_with_logits(outputs, targets.to(device))
                epoch_loss += loss.item() * loader.batch_size

                fin_targets.extend(targets.tolist())
                fin_probabs.extend(torch.sigmoid(outputs).detach().cpu().tolist())

             
            loss = epoch_loss / len(loader.dataset)
            fin_preds = np.rint(np.array(fin_probabs))

        return fin_preds, fin_probabs, fin_targets, full_hadm_ids, full_attn_weights, loss


def save_predictions(probabs, targets, dtset, hype):
    np.savetxt(f'../results/{dtset}_probabs_{hype}.txt', probabs)
    np.savetxt(f'../results/{dtset}_targets_{hype}.txt', targets)


def precision_at_k(true_labels, pred_probs):
    # num true labels in top k predictions / k
    ks = [1, 5, 8, 10, 15]
    sorted_pred = np.argsort(pred_probs)[:, ::-1]
    output = []
    p5_scores = None
    for k in ks:
        topk = sorted_pred[:, :k]

        # get precision at k for each example
        vals = []
        for i, tk in enumerate(topk):
            if len(tk) > 0:
                num_true_in_top_k = true_labels[i, tk].sum()
                denom = len(tk)
                vals.append(num_true_in_top_k / float(denom))

        output.append(np.mean(vals))
        if k == 5:
            p5_scores = np.array(vals)
    return output, p5_scores


def roc_auc(true_labels, pred_probs, average="macro"):
    if pred_probs.shape[0] <= 1:
        return

    fpr = {}
    tpr = {}
    if average == "macro":
        # get AUC for each label individually
        relevant_labels = []
        auc_labels = {}
        for i in range(true_labels.shape[1]):
            # only if there are true positives for this label
            if true_labels[:, i].sum() > 0:
                fpr[i], tpr[i], _ = roc_curve(true_labels[:, i], pred_probs[:, i])
                if len(fpr[i]) > 1 and len(tpr[i]) > 1:
                    auc_score = auc(fpr[i], tpr[i])
                    if not np.isnan(auc_score):
                        auc_labels["auc_%d" % i] = auc_score
                        relevant_labels.append(i)

        # macro-AUC: just average the auc scores
        aucs = []
        for i in relevant_labels:
            aucs.append(auc_labels['auc_%d' % i])
        score = np.mean(aucs)
    else:
        # micro-AUC: just look at each individual prediction
        flat_pred = pred_probs.ravel()
        fpr["micro"], tpr["micro"], _ = roc_curve(true_labels.ravel(), flat_pred)
        score = auc(fpr["micro"], tpr["micro"])

    return score


def micro_f1(true_labels, pred_labels):
    prec = micro_precision(true_labels, pred_labels)
    rec = micro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def micro_precision(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    if flat_pred.sum(axis=0) == 0:
        return 0.0
    return intersect_size(flat_true, flat_pred, 0) / flat_pred.sum(axis=0)


def micro_recall(true_labels, pred_labels):
    flat_true = true_labels.ravel()
    flat_pred = pred_labels.ravel()
    return intersect_size(flat_true, flat_pred, 0) / flat_true.sum(axis=0)


def intersect_size(x, y, axis):
    return np.logical_and(x, y).sum(axis=axis).astype(float)


def macro_recall(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (true_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def macro_f1(true_labels, pred_labels):
    prec = macro_precision(true_labels, pred_labels)
    rec = macro_recall(true_labels, pred_labels)
    if prec + rec == 0:
        f1 = 0.
    else:
        f1 = 2 * (prec * rec) / (prec + rec)
    return prec, rec, f1


def macro_precision(true_labels, pred_labels):
    num = intersect_size(true_labels, pred_labels, 0) / (pred_labels.sum(axis=0) + 1e-10)
    return np.mean(num)


def compute_scores(probabs, targets, hyper_params, dtset, alpha=0.03, full_hadm_ids=None, full_attn_weights=None, simi=False):
    args = constants.get_args()
    if simi:
        probabs = np.array(probabs)
        targets = np.array(targets)

        preds = np.zeros(probabs.shape)
        preds[np.where(probabs > alpha)] = 1

        print('start accuracy')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        accuracy = accuracy_score(targets, preds)

        print('start f1_score')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        _, _, f1_score_macro = macro_f1(targets, preds)
        _, _, f1_score_micro = micro_f1(targets, preds)

        print('start AUC')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        auc_score_micro = roc_auc(targets, preds, average='micro')
        auc_score_macro = roc_auc(targets, preds, average='macro')

        print('start precision')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        precision_at_ks, p5_scores = precision_at_k(targets, preds)

        logging.info(f"{dtset} Accuracy: {accuracy}")
        logging.info(f"{dtset} f1 score (micro): {f1_score_micro}")
        logging.info(f"{dtset} f1 score (macro): {f1_score_macro}")
        logging.info(f"{dtset} auc score (micro): {auc_score_micro}")
        logging.info(f"{dtset} auc score (macro): {auc_score_macro}")
        logging.info(f"{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")

        print(f"\n{dtset} accuracy: {accuracy}"
              f"\n{dtset} f1 score (micro): {f1_score_micro}"
              f"\n{dtset} f1 score (macro): {f1_score_macro}"
              f"\n{dtset} auc score (micro): {auc_score_micro}"
              f"\n{dtset} auc score (macro): {auc_score_macro}"
              f"\n{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}")

    else:
        probabs = np.array(probabs)
        targets = np.array(targets)

        preds = np.rint(probabs)  # (probabs >= 0.5)

        print('start accuracy')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        accuracy = accuracy_score(targets, preds)

        print('start f1 score')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        _, _, f1_score_micro = micro_f1(targets, preds)
        _, _, f1_score_macro = macro_f1(targets, preds)

        print('start AUC')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        auc_score_micro = roc_auc(targets, probabs, average='micro')
        auc_score_macro = roc_auc(targets, probabs, average='macro')

        print('start precision')
        print(datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        precision_at_ks, p5_scores = precision_at_k(targets, probabs)

        logging.info(f"{dtset} Accuracy: {accuracy}")
        logging.info(f"{dtset} f1 score (micro): {f1_score_micro}")
        logging.info(f"{dtset} f1 score (macro): {f1_score_macro}")
        logging.info(f"{dtset} auc score (micro): {auc_score_micro}")
        logging.info(f"{dtset} auc score (macro): {auc_score_macro}")
        logging.info(f"{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")

        print(f"\n{dtset} accuracy: {accuracy}"
              f"\n{dtset} f1 score (micro): {f1_score_micro}"
              f"\n{dtset} f1 score (macro): {f1_score_macro}"
              f"\n{dtset} auc score (micro): {auc_score_micro}"
              f"\n{dtset} auc score (macro): {auc_score_macro}"
              f"\n{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}")
        
        
        if args.data_setting == 'full':
            
            print(f"testing on rare codes...")

            probabs = np.array(probabs)[:, 3528:7015]
            targets = np.array(targets)[:, 3528:7015]

            preds = np.rint(probabs)  # (probabs >= 0.5)

            accuracy = accuracy_score(targets, preds)

            _, _, f1_score_micro = micro_f1(targets, preds)
            _, _, f1_score_macro = macro_f1(targets, preds)

            auc_score_micro = roc_auc(targets, probabs, average='micro')
            auc_score_macro = roc_auc(targets, probabs, average='macro')

            precision_at_ks, p5_scores = precision_at_k(targets, probabs)

            logging.info(f"{dtset} Rare Accuracy: {accuracy}")
            logging.info(f"{dtset} Rare f1 score (micro): {f1_score_micro}")
            logging.info(f"{dtset} Rare f1 score (macro): {f1_score_macro}")
            logging.info(f"{dtset} Rare auc score (micro): {auc_score_micro}")
            logging.info(f"{dtset} Rare auc score (macro): {auc_score_macro}")
            logging.info(f"{dtset} Rare precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}\n")

            print(f"\n{dtset} accuracy: {accuracy}"
                  f"\n{dtset} f1 score (micro): {f1_score_micro}"
                  f"\n{dtset} f1 score (macro): {f1_score_macro}"
                  f"\n{dtset} auc score (micro): {auc_score_micro}"
                  f"\n{dtset} auc score (macro): {auc_score_macro}"
                  f"\n{dtset} precision at ks [1, 5, 8, 10, 15]: {precision_at_ks}")