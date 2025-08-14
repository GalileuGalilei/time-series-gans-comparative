"""Reimplement TimeGAN-pytorch Codebase.

Reference: Jinsung Yoon, Daniel Jarrett, Mihaela van der Schaar,
"Time-series Generative Adversarial Networks,"
Neural Information Processing Systems (NeurIPS), 2019.

Paper link: https://papers.nips.cc/paper/8789-time-series-generative-adversarial-networks

Last updated Date: October 18th 2021
Code author: Zhiwei Zhang (bitzzw@gmail.com), Biaolin Wen(robinbg@foxmail.com)

-----------------------------

timegan.py

Note: Use original data as training set to generater synthetic data (time-series)
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from .data import batch_generator
from .utils import extract_time, random_generator, NormMinMax
from .model import Encoder, Recovery, Generator, Discriminator, Supervisor


class BaseModel():
  """ Base Model for timegan
  """

  def __init__(self, opt, ori_data, labels):
    # Seed for deterministic behavior
    self.seed(opt.manualseed)

    # Initalize variables.
    self.opt = opt
    self.ori_data, self.min_val, self.max_val = NormMinMax(ori_data)
    self.labels = labels
    self.ori_time, self.max_seq_len = extract_time(self.ori_data)
    self.trn_dir = os.path.join(self.opt.outf, self.opt.name, 'train')
    self.tst_dir = os.path.join(self.opt.outf, self.opt.name, 'test')
    self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

  def seed(self, seed_value):
    """ Seed

    Arguments:
        seed_value {int} -- [description]
    """
    # Check if seed is default value
    if seed_value == -1:
      return

    # Otherwise seed all functionality
    import random
    random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    np.random.seed(seed_value)
    torch.backends.cudnn.deterministic = True

  def save_weights(self, epoch):
    """Save net weights for the current epoch.

    Args:
        epoch ([int]): Current epoch number.
    """

    weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
    if not os.path.exists(weight_dir): 
      os.makedirs(weight_dir)

    torch.save({'epoch': epoch + 1, 'state_dict': self.nete.state_dict()},
               '%s/netE.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netr.state_dict()},
               '%s/netR.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
               '%s/netG.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
               '%s/netD.pth' % (weight_dir))
    torch.save({'epoch': epoch + 1, 'state_dict': self.nets.state_dict()},
               '%s/netS.pth' % (weight_dir))


  def train_one_iter_er(self):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    # set mini-batch
    self.X0, self.T, self.Y = batch_generator(self.ori_data, self.labels, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Y = torch.tensor(self.Y, dtype=torch.long).to(self.device)

    # train encoder & decoder
    self.optimize_params_er()

  def train_one_iter_er_(self):
    """ Train the model for one epoch.
    """

    self.nete.train()
    self.netr.train()

    self.X0, self.T, self.Y = batch_generator(self.ori_data, self.labels, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Y = torch.tensor(self.Y, dtype=torch.long).to(self.device)

    # train encoder & decoder
    self.optimize_params_er_()
 
  def train_one_iter_s(self):
    """ Train the model for one epoch.
    """

    #self.nete.eval()
    self.nets.train()

    self.X0, self.T, self.Y = batch_generator(self.ori_data, self.labels, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Y = torch.tensor(self.Y, dtype=torch.long).to(self.device)
    
    # train superviser
    self.optimize_params_s()

  def train_one_iter_g(self):
    """ Train the model for one epoch.
    """

    """self.netr.eval()
    self.nets.eval()
    self.netd.eval()"""
    self.netg.train()

    # set mini-batch
    self.X0, self.T, self.Y = batch_generator(self.ori_data, self.labels, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z, _ = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_g()

  def train_one_iter_d(self):
    """ Train the model for one epoch.
    """
    """self.nete.eval()
    self.netr.eval()
    self.nets.eval()
    self.netg.eval()"""
    self.netd.train()

    # set mini-batch
    self.X0, self.T, self.Y = batch_generator(self.ori_data, self.labels, self.ori_time, self.opt.batch_size)
    self.X = torch.tensor(self.X0, dtype=torch.float32).to(self.device)
    self.Z, _ = random_generator(self.opt.batch_size, self.opt.z_dim, self.T, self.max_seq_len)

    # train superviser
    self.optimize_params_d()


  def train(self):
    """ Train the model
    """

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_er()

      print('Encoder training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # Train for one iter
      self.train_one_iter_s()

      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))

    for iter in range(self.opt.iteration):
      # Train for one iter
      #for kk in range(2):
      self.train_one_iter_d()
      self.train_one_iter_g()
      self.train_one_iter_er_()


      print('Superviser training step: '+ str(iter) + '/' + str(self.opt.iteration))

    self.save_weights(self.opt.iteration)
    #self.generated_data = self.generation(self.opt.batch_size)
    print('Finish Synthetic Data Generation')

  """def evaluation(self):
    ## Performance metrics
    # Output initialization
    metric_results = dict()

    # 1. Discriminative Score
    discriminative_score = list()
    for _ in range(self.opt.metric_iteration):
      temp_disc = discriminative_score_metrics(self.ori_data, self.generated_data)
      discriminative_score.append(temp_disc)

    metric_results['discriminative'] = np.mean(discriminative_score)

    # 2. Predictive score
    predictive_score = list()
    for tt in range(self.opt.metric_iteration):
      temp_pred = predictive_score_metrics(self.ori_data, self.generated_data)
      predictive_score.append(temp_pred)

    metric_results['predictive'] = np.mean(predictive_score)

    # 3. Visualization (PCA and tSNE)
    visualization(self.ori_data, self.generated_data, 'pca')
    visualization(self.ori_data, self.generated_data, 'tsne')

    ## Print discriminative and predictive scores
    print(metric_results)
"""

  def generation(self, labels):
    all_samples = []
    batch_size = self.opt.batch_size
    n_batches = int(np.ceil(len(labels) / batch_size))
    Z = random_generator(len(labels), self.opt.z_dim, [self.max_seq_len for _ in range(len(labels))], self.max_seq_len)[0]
    #print(np.array([Z]).shape)

    for i in range(n_batches):
        current_labels = labels[i * batch_size:(i + 1) * batch_size]

        # Gera Z com tamanho correto
        Z_batch = Z[i * batch_size:(i + 1) * batch_size]
        Z_batch = torch.tensor(Z_batch, dtype=torch.float32).to(self.device)
        # Gera Y com tamanho correto
        Y_batch = torch.tensor(current_labels, dtype=torch.long).to(self.device)
        
        # último batch pode ser menor que o tamanho do batch
        if len(Z_batch) < batch_size:
            pad_size = batch_size - len(Z_batch)
            pad_shape = (pad_size, Z_batch.shape[1], Z_batch.shape[2])
            Z_batch = torch.cat([Z_batch, torch.zeros(pad_shape, device=Z_batch.device, dtype=Z_batch.dtype)], dim=0)
            Y_batch = torch.cat([Y_batch, torch.zeros(pad_size, device=Y_batch.device, dtype=Y_batch.dtype)], dim=0)

        # Gera sequências
        E_hat = self.netg(Z_batch, Y_batch)
        H_hat = self.nets(E_hat)
        X_hat = self.netr(H_hat).cpu().detach().numpy()
        
        #desnormaliza
        X_hat = X_hat * self.max_val + self.min_val

        all_samples.append(X_hat)

    generated = np.concatenate(all_samples, axis=0)
    return generated[:len(labels)]



class TimeGAN(BaseModel):
    """TimeGAN Class
    """

    @property
    def name(self):
      return 'TimeGAN'

    def __init__(self, opt, ori_data, labels):
      super(TimeGAN, self).__init__(opt, ori_data, labels)

      # -- Misc attributes
      self.epoch = 0
      self.times = []
      self.total_steps = 0

      # Create and initialize networks.
      self.nete = Encoder(self.opt).to(self.device)
      self.netr = Recovery(self.opt).to(self.device)
      self.netg = Generator(self.opt).to(self.device)
      self.netd = Discriminator(self.opt).to(self.device)
      self.nets = Supervisor(self.opt).to(self.device)

      if self.opt.resume != '':
        print("\nLoading pre-trained networks.")
        self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
        self.nete.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netE.pth'))['state_dict'])
        self.netr.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netR.pth'))['state_dict'])
        self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
        self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
        self.nets.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netS.pth'))['state_dict'])
        print("\tDone.\n")

      # loss
      self.l_mse = nn.MSELoss()
      self.l_r = nn.L1Loss()
      self.l_ce = nn.CrossEntropyLoss()
      self.l_bce = nn.BCELoss()

      # Setup optimizer
      if self.opt.isTrain:
        self.nete.train()
        self.netr.train()
        self.netg.train()
        self.netd.train()
        self.nets.train()
        self.optimizer_e = optim.Adam(self.nete.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_r = optim.Adam(self.netr.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        self.optimizer_s = optim.Adam(self.nets.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))


    def forward_e(self):
      """ Forward propagate through netE
      """
      self.H = self.nete(self.X)

    def forward_er(self):
      """ Forward propagate through netR
      """
      self.H = self.nete(self.X)
      self.X_tilde = self.netr(self.H)

    def forward_g(self):
      """ Forward propagate through netG
      """
      self.Z = torch.tensor(self.Z, dtype=torch.float32).to(self.device)
      self.Y = torch.tensor(self.Y, dtype=torch.long).to(self.device)
      self.E_hat = self.netg(self.Z, self.Y)

    def forward_rg(self):
      """ Forward propagate through netG
      """
      self.X_hat = self.netr(self.H_hat)

    def forward_s(self):
      """ Forward propagate through netS
      """
      self.H_supervise = self.nets(self.H)
      # print(self.H, self.H_supervise)

    def forward_sg(self):
      """ Forward propagate through netS
      """
      self.H_hat = self.nets(self.E_hat)



    def backward_er(self):
      """ Backpropagate through netE
      """
      self.err_er = self.l_mse(self.X_tilde, self.X)
      self.err_er.backward(retain_graph=True)
      print("Loss: ", self.err_er)

    def backward_er_(self):
      """ Backpropagate through netE
      """
      self.err_er_ = self.l_mse(self.X_tilde, self.X) 
      self.err_s = self.l_mse(self.H_supervise[:,:-1,:], self.H[:,1:,:])
      self.err_er = 5 * torch.sqrt(self.err_er_) + 0.1 * self.err_s
      self.err_er.backward(retain_graph=True)

    #  print("Loss: ", self.err_er_, self.err_s)

    def forward_dg(self):
      """ Forward propagate fake data (gerador) no discriminador """
      self.D_fake, self.D_fake_cls = self.netd(self.H_hat)
      self.D_fake_e, self.D_fake_cls_e = self.netd(self.E_hat)

    def forward_d(self):
      """ Forward propagate real e fake data no discriminador """
      self.D_real, self.D_real_cls = self.netd(self.H)
      self.D_fake, self.D_fake_cls = self.netd(self.H_hat)
      self.D_fake_e, self.D_fake_cls_e = self.netd(self.E_hat)

    def backward_g(self):
      """ Backpropagate erro do gerador """
      # Adversarial Loss
      self.err_g_U = self.l_bce(self.D_fake, torch.ones_like(self.D_fake))
      self.err_g_U_e = self.l_bce(self.D_fake_e, torch.ones_like(self.D_fake_e))

      # Classification Loss (queremos que o discriminador preveja a classe correta das fakes)
      self.err_g_cls = self.l_ce(self.D_fake_cls, self.Y)
      self.err_g_cls_e = self.l_ce(self.D_fake_cls_e, self.Y)

      # Moment Matching Losses (como antes)
      std_real = torch.std(self.X, dim=(0, 1))
      std_fake = torch.std(self.X_hat, dim=(0, 1))
      self.err_g_V1 = torch.mean(torch.abs(torch.sqrt(std_fake + 1e-6) - torch.sqrt(std_real + 1e-6)))

      mean_real = torch.mean(self.X, dim=(0, 1))
      mean_fake = torch.mean(self.X_hat, dim=(0, 1))
      self.err_g_V2 = torch.mean(torch.abs(mean_fake - mean_real))

      # Total Generator Loss
      self.err_g = self.err_g_U + \
                  self.err_g_U_e * self.opt.w_gamma + \
                  self.err_g_V1 * self.opt.w_g + \
                  self.err_g_V2 * self.opt.w_g + \
                  self.err_g_cls + \
                  self.err_g_cls_e * self.opt.w_gamma

      self.err_g.backward(retain_graph=True)
      print("Loss G:", self.err_g.item())

    def backward_d(self):
      """ Backpropagate erro do discriminador """
      # Adversarial losses
      self.err_d_real = self.l_bce(self.D_real, torch.ones_like(self.D_real))
      self.err_d_fake = self.l_bce(self.D_fake, torch.zeros_like(self.D_fake))
      self.err_d_fake_e = self.l_bce(self.D_fake_e, torch.zeros_like(self.D_fake_e))

      # Classification loss apenas para os reais
      self.err_d_cls = self.l_ce(self.D_real_cls, self.Y)

      # Total Discriminator Loss
      self.err_d = self.err_d_real + \
                  self.err_d_fake + \
                  self.err_d_fake_e * self.opt.w_gamma + \
                  self.err_d_cls

      if self.err_d > 0.15:
          self.err_d.backward(retain_graph=True)

      print("Loss D:", self.err_d.item())

    def backward_s(self):
      """ Backpropagate through netS
      """
      self.err_s = self.l_mse(self.H[:,1:,:], self.H_supervise[:,:-1,:])
      self.err_s.backward(retain_graph=True)
      print("Loss S: ", self.err_s)
   #   print(torch.autograd.grad(self.err_s, self.nets.parameters()))

    def optimize_params_er(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()

      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_er_(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_er()
      self.forward_s()
      # Backward-pass
      # nete & netr
      self.optimizer_e.zero_grad()
      self.optimizer_r.zero_grad()
      self.backward_er_()
      self.optimizer_e.step()
      self.optimizer_r.step()

    def optimize_params_s(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()

      # Backward-pass
      # nets
      self.optimizer_s.zero_grad()
      self.backward_s()
      self.optimizer_s.step()

    def optimize_params_g(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_s()
      self.forward_g()
      self.forward_sg()
      self.forward_rg()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_g.zero_grad()
      self.optimizer_s.zero_grad()
      self.backward_g()
      self.optimizer_g.step()
      self.optimizer_s.step()

    def optimize_params_d(self):
      """ Forwardpass, Loss Computation and Backwardpass.
      """
      # Forward-pass
      self.forward_e()
      self.forward_g()
      self.forward_sg()
      self.forward_d()
      self.forward_dg()

      # Backward-pass
      # nets
      self.optimizer_d.zero_grad()
      self.backward_d()
      self.optimizer_d.step()
