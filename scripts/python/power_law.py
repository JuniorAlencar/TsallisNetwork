import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['axes.linewidth'] = 1.4 #set the value globally

def p_r(r_max, d, alpha_g, r):
    alpha = alpha_g + d
    if alpha_g == 0.0:
        normalization_constant = 1 / (np.log(r_max) - np.log(1))
        return normalization_constant / (r ** d)
    else:
        normalization_constant = alpha_g * (r_max ** alpha_g / (r_max ** alpha_g - 1))
        return normalization_constant * r ** (-alpha)

r_max = 1000000
d = [1,2,3,4]
alpha_g = [0.0,1.0,2.0,3.0]

r_1 = np.linspace(1, 10, 1000)
r_2 = np.linspace(10, r_max, r_max)
r = np.concatenate((r_1, r_2))

pr_0_g = p_r(r_max, d[0], alpha_g[0], r)
pr_1_g = p_r(r_max, d[0], alpha_g[1], r)
pr_2_g = p_r(r_max, d[0], alpha_g[2], r)
pr_3_g = p_r(r_max, d[0], alpha_g[3], r)

# Normalize the distribution
norm_pr0_g = pr_0_g / np.sum(pr_0_g)
norm_pr1_g = pr_1_g / np.sum(pr_1_g)
norm_pr2_g = pr_2_g / np.sum(pr_2_g)
norm_pr3_g = pr_3_g / np.sum(pr_3_g)

pr_0_d = p_r(r_max, d[0], alpha_g[1], r)
pr_1_d = p_r(r_max, d[1], alpha_g[1], r)
pr_2_d = p_r(r_max, d[2], alpha_g[1], r)
pr_3_d = p_r(r_max, d[3], alpha_g[1], r)

# Normalize the distribution
norm_pr0_d = pr_0_d / np.sum(pr_0_d)
norm_pr1_d = pr_1_d / np.sum(pr_1_d)
norm_pr2_d = pr_2_d / np.sum(pr_2_d)
norm_pr3_d = pr_3_d / np.sum(pr_3_d)

color = ["#808080","darkgoldenrod",'#008000',"#00008B","magenta"]

labels_d = [f"$d = ${i}" for i in d]
labels_g = [f"$\\alpha_g = ${i}" for i in alpha_g]

fig, ax = plt.subplots(1,2,figsize=(19,8))
ax[0].plot(r,norm_pr0_d,label=labels_d[0],linewidth=2)
ax[0].plot(r,norm_pr1_d,label=labels_d[1],linewidth=2)
ax[0].plot(r,norm_pr2_d,label=labels_d[2],linewidth=2)
ax[0].plot(r,norm_pr3_d,label=labels_d[3],linewidth=2)
ax[0].set_title("(a)",fontsize=30)
ax[1].plot(r,norm_pr0_g,label=labels_g[0],linewidth=2)
ax[1].plot(r,norm_pr1_g,label=labels_g[1],linewidth=2)
ax[1].plot(r,norm_pr2_g,label=labels_g[2],linewidth=2)
ax[1].plot(r,norm_pr3_g,label=labels_g[3],linewidth=2)
ax[1].set_title("(b)",fontsize=30)
ax[0].set_ylabel(r"$P(r)$",size=21)
for i in range(2):
    #ax[i].set_xlim([0,8])
    ax[i].tick_params(which='minor', width=1.4, length=4,labelsize=20)
    ax[i].tick_params(which='major', width=1.4, length=8,labelsize=20)
    ax[i].legend(prop={"size":16},fancybox=True,framealpha=0.0,loc="lower left")
    ax[i].set_xlabel(r"$r$",size=21)
    ax[i].set_xscale("log")
    ax[i].set_yscale("log")
ax[0].text(10**(5.2), 10**(-2), f"$\\alpha_g = {alpha_g[1]}$", style="normal" ,fontsize=16, bbox={'facecolor': color[0], 'alpha': 0.1, 'pad': 10})
ax[1].text(10**(5.49), 10**(-2), f"$d = {d[0]}$", style="normal" ,fontsize=16, bbox={'facecolor': color[0], 'alpha': 0.1, 'pad': 10})

plt.savefig("../../results/log_p_r.pdf",dpi=300)

plt.show()

'''
labels = [f"$d$ = {i}" for i in d]
plt.figure(figsize=(10,10))





plt.xlim([0,10])
plt.ylabel(r"$P(r)$",fontsize=22)
plt.xlabel(r"$r$",fontsize=22)
plt.tick_params(axis='both', which='major', direction="in",width=1.4,length=8,labelsize=15)
plt.legend(fontsize=19,frameon=False)

ax = plt.gca()  # Obter o objeto da classe Axes
for spine in ax.spines.values():
    spine.set_linewidth(1.4)  # Ajuste a espessura da borda da caixa aqui

text = r"$\alpha_g = 2.0$"
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor="none")
plt.annotate(text, xy=(0.65, 0.92), xycoords='axes fraction', fontsize=19, ha="center", bbox=bbox_props)
plt.savefig("../../results/p_r_d.pdf",dpi=300)
#plt.show()
'''



'''
labels = [f"$\\alpha_g$ = {i}" for i in alpha_g]
plt.figure(figsize=(10,10))
print(sum(norm_pr3))
plt.plot(r,norm_pr0,label=labels[0],linewidth=2)
plt.plot(r,norm_pr1,label=labels[1],linewidth=2)
plt.plot(r,norm_pr2,label=labels[2],linewidth=2)
plt.plot(r,norm_pr3,label=labels[3],linewidth=2)
plt.xlim([0,20])
plt.ylabel(r"$P(r)$",fontsize=22)
plt.xlabel(r"$r$",fontsize=22)
plt.tick_params(axis='both', which='major', direction="in",width=1.4,length=8,labelsize=15)
plt.legend(fontsize=17,frameon=False)

ax = plt.gca()  # Obter o objeto da classe Axes
for spine in ax.spines.values():
    spine.set_linewidth(1.4)  # Ajuste a espessura da borda da caixa aqui
text = r"$d = 1.0$"
bbox_props = dict(boxstyle="round,pad=0.3", edgecolor="k", facecolor="none")
plt.annotate(text, xy=(0.65, 0.92), xycoords='axes fraction', fontsize=19, ha="center", bbox=bbox_props)
plt.savefig("../../results/p_r_alpha_g.pdf",dpi=300)
plt.show()
'''
