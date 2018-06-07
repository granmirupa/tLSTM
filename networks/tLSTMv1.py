from torch import nn;
import torch;
import torch.nn.functional as F;
from torch.autograd import Variable;

class tLSTMv1(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, batch_first=False, num_labels = 5):
        super(tLSTMv1, self).__init__()
        
        self.batch_first = batch_first
        
        self.hidden_size = hidden_size;
        self.input_size = input_size;
        self.output_size = output_size;
        self.num_labels = num_labels;
	
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )        
	
        # input embedding
        self.input_time = nn.Linear(1, self.num_labels);
       
        # lstm weights for previous labels 
        self.weight_fm = nn.Linear(self.num_labels, hidden_size) 		# 5 ==> 1024 
        self.weight_im = nn.Linear(self.num_labels, hidden_size) 		# 5 ==> 1024 
        self.weight_cm = nn.Linear(self.num_labels, hidden_size) 		# 5 ==> 1024 
        self.weight_om = nn.Linear(self.num_labels, hidden_size) 		# 5 ==> 1024 
        # lstm weights for features 
        self.weight_fx = nn.Linear(input_size, hidden_size) 			# 1024 ==> 1024 
        self.weight_ix = nn.Linear(input_size, hidden_size) 			# 1024 ==> 1024 
        self.weight_cx = nn.Linear(input_size, hidden_size) 			# 1024 ==> 1024 
        self.weight_ox = nn.Linear(input_size, hidden_size) 			# 1024 ==> 1024 
        # lstm weights for time features 
        self.weight_ft = nn.Linear(1, hidden_size) 				# 5 ==> 1024 
        self.weight_it = nn.Linear(1, hidden_size)		 		# 5 ==> 1024 
        self.weight_ct = nn.Linear(1, hidden_size) 				# 5 ==> 1024 
        self.weight_ot = nn.Linear(1, hidden_size)		 		# 5 ==> 1024 

        self.decoder = nn.Sequential(
		nn.Linear(hidden_size, hidden_size), 				# 1024 ==> 1024
		nn.Linear(hidden_size, self.num_labels),			# 1024 ==> 5
		nn.Sigmoid()
        )

    def forward(self, input, labels, time, ctx = None, ctx_mask=None):   
        def recurrence(input, labels_prev, time, hidden):
            """Recurrence helper."""
            
            # Previous state
            cx = hidden  # n_b x hidden_dim
            
            # features input
            features = input.view(1,-1);
	    
            #Gates 
            ingate = F.sigmoid(self.weight_ix(features) + self.weight_im(labels_prev) + self.weight_it(time))
            forgetgate = F.sigmoid(self.weight_fx(features) + self.weight_fm(labels_prev) + self.weight_ft(time))
            cellgate = F.tanh(self.weight_cx(features) + self.weight_cm(labels_prev) + self.weight_ct(time))
            outgate = F.sigmoid(self.weight_ox(features) + self.weight_om(labels_prev) + self.weight_ot(time))
            
            cy = (forgetgate * cx) + (ingate * cellgate)
            hy = outgate * F.tanh(cy)  # n_b x hidden_dim
            
	    hy = self.decoder(hy)

            return hy, cy

        if self.batch_first:
            input = input.transpose(0, 1)

        output = []
        steps = range(input.size(0))
        hidden = self.hidden;
	input = self.features(input); 

        for i in steps:
            hidden = recurrence(input[i], labels[i], time[i], hidden)
            if isinstance(hidden, tuple):
                output.append(hidden[0])
                hidden = hidden[1];
            else:
                output.append(hidden)
                
        output = torch.cat(output, 0).view(input.size(0), self.num_labels)

        if self.batch_first:
            output = output.transpose(0, 1)

        return output, hidden


    def init_hidden_(self):
       self.hidden = Variable(torch.zeros(1,self.hidden_size).cuda());
