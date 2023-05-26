from random import seed
# from webicd.utils import is_visible_luv_gama
from webicd.utils import color_distance, is_visible_luv_gama
from webicd.icd import differentiantion_ar, differentiantion_mean
from .abstract import AbstractRecolorConfusionColorStrategy
import pygad


class GeneticRecolor(AbstractRecolorConfusionColorStrategy):
    def __init__(self, ellipse, confusion_point, luminances, random_state=None) -> None:
        self.ellipse = ellipse
        self.confusion_point = confusion_point
        self.luminances = luminances
        self.random_state = random_state

    def fitness(self, original_color, colors_array, alpha):
        def fitness_func(solution, solution_idx):
            diff = differentiantion_mean(
                self.ellipse,
                self.confusion_point,
                self.luminances,
                solution, colors_array
            )
            distance = color_distance(original_color, solution)
            
            visible = 0 if is_visible_luv_gama(solution) else -10000000
            return (alpha * diff) + ((1 - alpha) * -distance) + visible

        return fitness_func

    def get_GA(self, original_color, colors_array, alpha):
        seed(self.random_state)
        fitness_function = self.fitness(original_color, colors_array, alpha)
        num_generations = 1000
        num_parents_mating = 4

        sol_per_pop = 20
        num_genes = 3

        init_range_low = 0
        init_range_high = 1

        parent_selection_type = "sss"
        keep_parents = 1

        crossover_type = "single_point"

        mutation_type = "random"
        mutation_num_genes = 3
        mutation_by_replacement = False

        gene_space = [{'low': 0, 'high': 100}, {
            'low': 0, 'high': 0.7}, {'low': 0, 'high': 0.6}]

        ga_instance = pygad.GA(num_generations=num_generations,
                               num_parents_mating=num_parents_mating,
                               fitness_func=fitness_function,
                               sol_per_pop=sol_per_pop,
                               num_genes=num_genes,
                               init_range_low=init_range_low,
                               init_range_high=init_range_high,
                               parent_selection_type=parent_selection_type,
                               keep_parents=keep_parents,
                               crossover_type=crossover_type,
                               mutation_type=mutation_type,
                               mutation_num_genes=mutation_num_genes,
                               gene_space=gene_space,
                               mutation_by_replacement=mutation_by_replacement)
        return ga_instance

    def execute(self, colorGraph):
        color_dict = {}
        colors = colorGraph.vs["color"]

        # while there is some node with degree greater than 0
        while len(colorGraph.vs(_degree_gt=0)) > 0:
            vertices = colorGraph.vs.select(_degree=colorGraph.maxdegree())
            v = vertices[0]
            v_color = v["color"]
            index = v.index
            # print(v_color)
            ga_instance = self.get_GA(
                v_color, colors[:index]+colors[index+1:], 0.5)
            ga_instance.run()
            solution, solution_fitness, solution_idx = ga_instance.best_solution()
            print("Parameters of the best solution : {solution}".format(
                solution=solution))
            print("Fitness value of the best solution = {solution_fitness}".format(
                solution_fitness=solution_fitness))
            print("Distance between original and best solution = {solution_distance}".format(
                solution_distance=color_distance(v_color, solution)
            ))
            print("Difference between solution and other colors = {solution_difference}".format(
                solution_difference=differentiantion_mean(self.ellipse,
                                                          self.confusion_point,
                                                          self.luminances,
                                                          solution, colors[:index]+colors[index+1:])
            ))

            colorGraph.delete_edges(v.all_edges())
            color_dict[v_color] = tuple(solution)

        for vertice in colorGraph.vs:
            color = vertice["color"]
            if color_dict.get(color) is None:
                color_dict[vertice["color"]] = vertice["color"]

        return color_dict
