def heatmap(dataframe, propertie = "short_mean", label_prop = r'$\langle \ell \rangle$' ,dim = 1):
        # Create heatmap
        pivot_table = dataframe.pivot_table(index='alpha_a', columns='alpha_g', values=propertie)

        # Fator de escala do tamanho dos elementos do gráfico
        fator = 2

        # Define o tamanho do plot
        plt.figure(figsize=(10,8))

        #sns.heatmap(data=pivot_table, cmap='YlOrBr', yticklabels=alpha_a_values)
        ax = sns.heatmap(data=pivot_table, cmap='YlOrBr')

        # Seleciona os valores de posições dos ticks
        loc_a = np.linspace(0, 100, 11)
        loc_g = np.linspace(0.1, 100.1, 11)

        # Indica os labels dos ticks selecionados
        labels_a = np.linspace(0, 10, 11)
        labels_g = np.linspace(0.1, 10.1, 11)

        # Set the tick positions
        plt.xticks(ticks=loc_g, labels=labels_g, rotation=0)
        plt.yticks(ticks=loc_a, labels=labels_a)

        plt.title(f'Dim = {dim}', fontsize=10*fator)
        plt.xlabel(r'$\alpha_g$', fontsize=9*fator)
        plt.ylabel(r'$\alpha_a$', fontsize=9*fator)

        ax.tick_params(axis='both', which='major', labelsize=6*fator)

        # Color bar
        cbar = ax.collections[0].colorbar
        cbar.set_label(label_prop, fontsize=9*fator)
        cbar.ax.tick_params(axis='both', which='major', labelsize=6*fator)

        plt.tight_layout()
        plt.show()

        plt.show()

heatmap(df_, propertie = "ass_coeff_mean")