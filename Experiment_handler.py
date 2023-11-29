import numpy as np
import pandas as pd

class Experiment_handler:
    def __init__(self, model_h, batch_size=64, experiment_name=None):
        self.model = model_h
        self.batch_size = batch_size
        #self.canvas_size = self.model.canvas_size
        self.experiment_name = experiment_name
        
    def train(self, epochs, data_h_train, data_h_val):
        lowest_loss = {'dice':-1,'epoch':-1}
        for epoch in range(epochs):
            d_l_fg_c = {0:[],1:[],2:[],3:[],4:[],5:[]}
            d_l_bg_c = {0:[],1:[],2:[],3:[],4:[],5:[]}
            loss = []
            ce = []
            for batch in data_h_train.get_batch(self.batch_size):
                imgs = np.array([b[0] for b in batch])
                segs = np.array([b[1] for b in batch], dtype=np.int32)
                keys = [b[2] for b in batch]

                out = self.model.model_true.train_on_batch(imgs, segs, return_dict=True, reset_metrics=True)
                loss.append(out['loss'])
                ce.append(out['ce'])
                for key, bg, fg  in zip(keys, out['arr_d_bg'], out['arr_d_fg']):
                    d_l_bg_c[key].append(bg)
                    d_l_fg_c[key].append(fg)
            print('~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
            print('Done epoch:',epoch)
            total_bg = []
            total_fg = []
            for key in d_l_fg_c:
                print('avg dice loss for',key,'is: [',np.mean(d_l_bg_c[key]),',',np.mean(d_l_fg_c[key]),']')
                if key < 5:
                    total_bg.append(d_l_bg_c[key])
                    total_fg.append(d_l_fg_c[key])
            total_bg = np.concatenate(total_bg)
            total_fg = np.concatenate(total_fg)
            print('avg total loss:',np.mean(loss))
            print('avg total ce:',np.mean(ce))
            print('avg dice loss is: [',np.mean(total_bg),',',np.mean(total_fg),']')
                
                
                
            #Val code
            d_l_fg_c_v = {0:[],1:[],2:[],3:[],4:[],5:[]}
            d_l_bg_c_v = {0:[],1:[],2:[],3:[],4:[],5:[]}
            loss_v = []
            ce_v = []
            for batch in data_h_val.get_batch(self.batch_size):
                imgs = np.array([b[0] for b in batch])
                segs = np.array([b[1] for b in batch], dtype=np.int32)
                keys = [b[2] for b in batch]
                
                out = self.model.model_false.test_on_batch(imgs, segs, return_dict=True, reset_metrics=True) #Does this account for batch norm?
                loss_v.append(out['loss'])
                ce_v.append(out['ce'])
                for key, bg, fg  in zip(keys, out['arr_d_bg'], out['arr_d_fg']):
                    d_l_bg_c_v[key].append(bg)
                    d_l_fg_c_v[key].append(fg)
            print('Done TEST')
            total_bg_v = []
            total_fg_v = []
            for key in d_l_fg_c_v:
                print('avg dice test loss for',key,'is: [',np.mean(d_l_bg_c_v[key]),',',np.mean(d_l_fg_c_v[key]),']')
                if key < 5:
                    total_bg_v.append(d_l_bg_c_v[key])
                    total_fg_v.append(d_l_fg_c_v[key])
            total_bg_v = np.concatenate(total_bg_v)
            total_fg_v = np.concatenate(total_fg_v)
            print('avg total test loss:',np.mean(loss_v))
            print('avg total test ce:',np.mean(ce_v))
            avg_d_fg = np.mean(total_fg_v)
            print('avg test dice loss is: [',np.mean(total_bg_v),',',avg_d_fg,']')
            if avg_d_fg > lowest_loss['dice']:
                if avg_d_fg > 0.63:# and epoch > 10:
                    print('!!!!Saved model!!!!')
                    self.model.model_true.save_weights('./saved_weights/dice_'+str(avg_d_fg)+'_e'+str(epoch))
                lowest_loss['dice'] = avg_d_fg
                lowest_loss['epoch'] = epoch
            print('Max total dice is',lowest_loss['dice'],'at epoch',lowest_loss['epoch'])
    
    def evaluate_dice(self, pred, gt):
        num = np.sum(2*pred*gt)
        dem = np.sum(pred+gt)
        return num/dem
    
    def test(self, data_h_test, tort='train'):
        output = pd.DataFrame(columns=['id','rle'])
        total_dice = []
        c = 0
        for batch in data_h_test.get_test_img(tort):
            if c%100 == 0 and tort == 'train':
                print(c)
                #pass
            c += 1
            img = batch[0]
            shape = batch[1]
            ps = batch[2]
            img_id = batch[3]
            seg_gt = batch[4]
            
            in_img = np.array([img])
            seg_f = self.model.call_net(in_img, training=False).numpy()[0,:,:,1]
                             
            seg = seg_f
            seg[seg > 0.5] = 1
            seg[seg < 0.6] = 0
            seg = data_h_test.convert_from_canvas(seg[:,:,np.newaxis], shape, ps)#[:,:,0]
            
            #print('dice:',d)
            
            if tort == 'train':
                d = self.evaluate_dice(seg, seg_gt)
                #print('Dice value:',d)
                total_dice  += [d]
            
            
            seg = np.array(seg, dtype=np.uint8)
                             
            output = output.append({'id': img_id}, ignore_index=True)
        if tort == 'train':
            print('Len tested:',len(total_dice))
            print('avg dice:',np.mean(total_dice))
        output.to_csv('./submission.csv', index=False)
        #print(output)