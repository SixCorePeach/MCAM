import torch
from fairscale.nn.misc import checkpoint_wrapper
import torch.nn as nn
from src.modeling.load_sensor_pred_head import get_sensor_pred_model
from src.modeling.3DCNN.CNN import cnn
# this file is the core of whole model ADAPT
class  causal(torch.nn.Module):
    def __init__(self, in_size, mid_size, out_size):
        super(causal, self).__init__()
        """
            this module will get four information including the env, the neigbor1, neigbor2, history, the history + neigbor1,2 to get action, and the env + neigbor 1,2 is reason to action  
            if this module add the dropout, then it could be improved more.

        """
        self.layersize = [512, 256, 512]
        self.MLP_history = nn.Sequential(nn.Linear(in_size, self.layersize[0]),
                nn.GELU(),
                nn.Linear(self.layersize[0], self.layersize[1]),
                nn.GELU(),
                nn.Linear(self.layersize[1], self.layersize[2]),
                nn.GELU(),
                nn.Linear(self.layersize[2], out_size),
                nn.GELU(),
        )

        self.MLP_env = nn.Sequential(nn.Linear(in_size, self.layersize[0]),
                nn.GELU(),
                nn.Linear(self.layersize[0], self.layersize[1]),
                nn.GELU(),
                nn.Linear(self.layersize[1], self.layersize[2]),
                nn.GELU(),
                nn.Linear(self.layersize[2], out_size),
                nn.GELU(),
        )


        self.MLP_neighbor = nn.Sequential(nn.Linear(in_size, self.layersize[0]),
                nn.GELU(),
                nn.Linear(self.layersize[0], self.layersize[1]),
                nn.GELU(),
                nn.Linear(self.layersize[1], self.layersize[2]),
                nn.GELU(),
                nn.Linear(self.layersize[2], out_size),
                nn.GELU(),
        )


        self.MLP_neighbor1 = nn.Sequential(nn.Linear(in_size, self.layersize[0]),
                nn.GELU(),
                nn.Linear(self.layersize[0], self.layersize[1]),
                nn.GELU(),
                nn.Linear(self.layersize[1], self.layersize[2]),
                nn.GELU(),
                nn.Linear(self.layersize[2], out_size),
                nn.GELU(),
        )


        self.MLP_condiction_1 = nn.Sequential(nn.Linear(3 * out_size, out_size),
                nn.GELU(),
                nn.Linear(out_size, out_size // 2),
                nn.GELU(),
                nn.Linear(out_size // 2, out_size // 8),
                nn.GELU(),
                nn.Linear(out_size // 8, 1),
                nn.Softmax(dim=1)
        )
        self.MLP_condiction_2 = nn.Sequential(nn.Linear(3 * mid_size, mid_size),
                nn.GELU(),
                nn.Linear(mid_size, mid_size//4),
                nn.GELU(),
                nn.Linear(mid_size//4, mid_size//16),
                nn.GELU(),
                nn.Linear(mid_size//16, 1),
                nn.Softmax(dim=2)
        )

        
    def forward(self, vid_feats):
        
        
        c0 = self.MLP_history(vid_feats)
        c1 = self.MLP_neighbor(vid_feats)
        c2 = self.MLP_neighbor1(vid_feats)
        c3 = self.MLP_env(vid_feats)


        c = torch.cat([c0, c1, c2],dim=2)
        c1 = torch.cat([c1, c2, c3],dim=1).permute(0, 2, 1)
        y1 = self.MLP_condiction_1(c)
        y2 = self.MLP_condiction_2(c1).permute(0, 2, 1)

        out = y1  @ y2 * vid_feats  + vid_feats
        return out
        

class MultitaskVideoTransformer(torch.nn.Module):
    """ This is the multi-task module that performs Driving Caption Generation and Control Signal Prediction. """
    """
        there will use the 3DCNN as the other encoder to give the feature for bert

    """

    def __init__(self, args, config, swin, transformer_encoder):
        """ Initializes the model.
        Parameters:
            args: basic args of ADAPT, mostly defined in `src/configs/VidSwinBert/BDDX_multi_default.json` and input args
            config: config of transformer_encoder, mostly defined in `models/captioning/bert-base-uncased/config.json`
            swin: torch module of the backbone to be used. See `src/modeling/load_swin.py`
            transformer_encoder: torch module of the transformer architecture. See `src/modeling/load_bert.py`
        """
        super(MultitaskVideoTransformer, self).__init__()
        self.config = config
        self.use_checkpoint = args.use_checkpoint and not args.freeze_backbone
        if self.use_checkpoint:
            self.swin = checkpoint_wrapper(swin, offload_to_cpu=True)
        else:
            self.swin = swin
        self.CNN = cnn('resnext101')
        self.trans_encoder = transformer_encoder # bert
        self.img_feature_dim = int(args.img_feature_dim)
        self.use_grid_feat = args.grid_feat
        self.latent_feat_size = self.swin.backbone.norm.normalized_shape[0]
        self.fc = nn.Linear(self.latent_feat_size, self.img_feature_dim)

        # the number of raodong up top could be compute by the LDU decomposed

        self.compute_mask_on_the_fly = False # deprecated
        self.mask_prob = args.mask_prob
        self.mask_token_id = -1
        self.max_img_seq_length = args.max_img_seq_length

        # get Control Signal Prediction Head
        self.sensor_pred_head = get_sensor_pred_model(args)

        # if only_signal is True, it means we 
        # remove Driving Caption Generation head and only use Control Signal Prediction head 
        self.only_signal = getattr(args, 'only_signal', False)
        self.vae = causal(512, 784, 512)
        # sparse attention mask defined in SwinBert #  the spare attention is defined ate here but the self casual attention is not found
        self.learn_mask_enabled = getattr(args, 'learn_mask_enabled', False)
        self.sparse_mask_soft2hard = getattr(args, 'sparse_mask_soft2hard', False)

        if self.learn_mask_enabled==True:
            self.learn_vid_att = torch.nn.Embedding(args.max_img_seq_length*args.max_img_seq_length,1)
            self.sigmoid = torch.nn.Sigmoid()

    def one_process(self, *args, **kwargs):

            # learn soft attention mask
            if self.learn_mask_enabled:
                kwargs['attention_mask'] = kwargs['attention_mask'].float() 
                vid_att_len = self.max_img_seq_length
                learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
                learn_att = self.sigmoid(learn_att)
                diag_mask = torch.diag(torch.ones(vid_att_len)).cuda() # the one diag matrix
                video_attention = (1. - diag_mask) * learn_att
                learn_att = diag_mask + video_attention
                if self.sparse_mask_soft2hard:
                    learn_att = (learn_att>=0.5) * 1.0
                    learn_att = learn_att.cuda()
                    learn_att.requires_grad = False
                kwargs['attention_mask'][:, -vid_att_len::, -vid_att_len::] = learn_att
                #  the sparse attention implement at here

            outputs = self.trans_encoder(*args, **kwargs)

            # outputs is the word end out, so we should do the work in bert but here
            # Control Signal Prediction head, output is ()

            sensor_outputs = self.sensor_pred_head(*args, **kwargs)
            return outputs, sensor_outputs, video_attention

    def forward(self, *args, **kwargs):
        """ The forward process of ADAPT, 
        Parameters:
            input_ids: word tokens of input sentences tokenized by tokenizer
            attention_mask: multimodal attention mask in Vision-Language transformer
            token_type_ids: typen tokens of input sentences, 
                            0 means it is a narration sentence and 1 means a reasoning sentence, same size with input_ids
            img_feats: preprocessed frames of the video
            masked_pos: [MASK] position when performing MLM, used to locate the masked words
            masked_ids: groung truth of [MASK] when performing MLM
            car_info: control signals of ego car in the video
        """

        # grad cam can only input a tuple (args, kwargs)
        if isinstance(args, tuple) and len(args) != 0:
            kwargs = args[0]
            args = ()

        # video swin to extract video features
        images = kwargs['img_feats']
        B, S, C, H, W = images.shape  # batch, segment, chanel, hight, width
        # (B x S x C x H x W) --> (B x C x S x H x W)
        images = images.permute(0, 2, 1, 3, 4)
        CNN_feats = self.CNN(images)
        print(CNN_feats.shape, 'CNN_feats')
        # the swin is video swin transformer, we do not change it

        vid_feats = self.swin(images)

        # tokenize video features to video tokens
        if self.use_grid_feat==True:
            vid_feats = vid_feats.permute(0, 2, 3, 4, 1)
        vid_feats = vid_feats.view(B, -1, self.latent_feat_size)

        # use an mlp to transform video token dimension


        vid_feats = self.fc(vid_feats)
        CNN_feats = self.fc(CNN_feats)

        # the mlp used to translate the  video token to adjust for vision language transformer
        #print(vid_feats.shape, 'stand')

        # prepare VL transformer inputs
        vid_feats = self.vae(vid_feats, CNN_feats)



        # disable bert attention outputs to avoid some bugs
        if self.trans_encoder.bert.encoder.output_attentions:
            self.trans_encoder.bert.encoder.set_output_attentions(False)
        
        if self.only_signal:
            # only Control Signal Prediction head 
            kwargs['img_feats'] = vid_feats
            sensor_outputs = self.sensor_pred_head(*args, **kwargs)        
            return sensor_outputs
        
        else:
            kwargs['img_feats'] = vid_feats
            outputs, sensor_outputs, video_attention = self.one_process(*args, **kwargs)
            outputs = outputs + sensor_outputs
            # sparse attention mask loss
            if self.learn_mask_enabled:
                loss_sparsity = self.get_loss_sparsity(video_attention)  
                outputs = outputs + (loss_sparsity, )
            return outputs
    
    def get_loss_sparsity(self, video_attention):
        sparsity_loss = 0
        sparsity_loss += (torch.mean(torch.abs(video_attention)))
        return sparsity_loss

    # orginal has three attn mask for 
    def reload_attn_mask(self, pretrain_attn_mask): 
        import numpy
        pretrained_num_tokens = int(numpy.sqrt(pretrain_attn_mask.shape[0]))

        pretrained_learn_att = pretrain_attn_mask.reshape(
                                pretrained_num_tokens,pretrained_num_tokens)
        scale_factor = 1
        vid_att_len = self.max_img_seq_length
        learn_att = self.learn_vid_att.weight.reshape(vid_att_len,vid_att_len)
        with torch.no_grad():
            for i in range(int(scale_factor)):
                learn_att[pretrained_num_tokens*i:pretrained_num_tokens*(i+1), 
                            pretrained_num_tokens*i:pretrained_num_tokens*(i+1)] = pretrained_learn_att 

    def freeze_backbone(self, freeze=True):
        for _, p in self.swin.named_parameters():
            p.requires_grad =  not freeze

