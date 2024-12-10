# 投骰子概率图的绘制
import torch
from torch.distributions import multinomial
from matplotlib import pyplot as plt
fair = torch.ones(6)
data = multinomial.Multinomial(100,fair).sample((500,))
cumdata = data.cumsum(dim=0)
finalData = cumdata / cumdata.sum(dim=1,keepdim=True)
for i in range(6):
    plt.plot(finalData[:,i],label='P(die=' + str(i+1) + ')')
plt.axhline(1/6,color='black',linestyle='dashed',label='base')
plt.legend()
plt.xlabel('Groups of experiments')
plt.ylabel('Estimated probability')
plt.title('precent')
plt.show()