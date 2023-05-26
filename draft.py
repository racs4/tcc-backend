# from colour import luminance
# from webicd.convert import luv_chroma_to_luv_gama
# from webicd.icd import differentiation


# PROTANOPE_CONFUSION = [50, 0.678, 0.501]
# DEUTERANOPE_CONFUSION = [50, -1.217, 0.782]
# TRITANOPE_CONFUSION = [50, 0.257, 0.0]
# a = [
#     PROTANOPE_CONFUSION,
#     DEUTERANOPE_CONFUSION,
#     TRITANOPE_CONFUSION,
# ]

# print(
#     list(
#         map(
#             lambda x: luv_chroma_to_luv_gama(x),
#             a
#         )
#     )
# )

# (312.10411386608496, 21.231403093933668)
# (-919.645886133915,203.8814030939337)
# (38.45411386608495, -304.41859690606634)


from scipy.stats import kruskal


def sum2(*a):
    return sum(a)

print(sum2(*[1,2,3,4]))
print(kruskal([1,2,3,4],[1,2,3,4]))