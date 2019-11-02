# -*- coding: utf-8 -*-
"""\
Small snippets and functions that were not so useful after all.
"""

#
# Part 1
#

# --------------------------------------------------------------------------------------------------

# Verify that the first 2 principal components form an orthonormal basis in the 1280-dimensional 
# space.

# comp_0 = F_pca_2d.components_[0]
# comp_1 = F_pca_2d.components_[1]
# display(
#     f'{np.linalg.norm(comp_0):.5f}',
#     f'{np.linalg.norm(comp_1):.5f}',
#     f'{np.dot(comp_0, comp_1):.5f}'
# )


# --------------------------------------------------------------------------------------------------

# Make a biplot of the 10 unit vectors with the longest projection.

# N_UNIT_VECTORS = 10

# # Make the biplot.
# fig, ax = make_2d_plot(F_rescaled_2d, y, names)

# # Create a data-frame sorted according to the norm of the projection.
# df_unit_projs = pd.DataFrame({
#     'proj-norm': np.linalg.norm(F_pca_2d.components_, axis=0),
#     'proj-comp-0': F_pca_2d.components_[0],
#     'proj-comp-1': F_pca_2d.components_[1]
# }).sort_values('proj-norm', ascending=False)

# # Plot the projections of the unit-vectors with the longest projections.
# for i in range(N_UNIT_VECTORS):
#     # Get weights.
#     weights = df_unit_projs.iloc[i]['proj-comp-0':'proj-comp-1'].to_numpy()
#     # Scale the weights.
#     weights = np.multiply(weights, 100)  # 100 chosen to make the plot more readable.
#     wx, wy = weights

#     # Plot arrow.
#     plt.arrow(
#         0, 0,  # Arrow starts at (0, 0)
#         wx, wy,  # ... and ends at (wx, wy).
#         color='black', width=0.2)

#     # Add text.
#     feature = f'{df_unit_projs.index[i]}'
#     text = plt.text(
#         wx * 1.5, # 1.5 chosen to make the plot more readable.
#         wy * 1.5,
#         feature,
#         weight='bold', color='white')

#     # Make the text stand out.
#     text.set_path_effects([
#         path_effects.Stroke(linewidth=2, foreground='black'),
#         path_effects.Normal()])

# plt.show()