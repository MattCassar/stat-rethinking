from causalgraphicalmodels import CausalGraphicalModel


def cgm():
    g = CausalGraphicalModel(
        nodes=['A', 'S', 'X', 'Y'],
        edges=[
            ('A', 'S'),
            ('A', 'X'),
            ('A', 'Y'),
            ('S', 'X'),
            ('S', 'Y'),
            ('X', 'Y'),
        ],
        latent_edges=[
            ('S', 'Y'),  # S <- U -> Y
        ]
    )
    g.draw().view()
    sets = g.get_all_backdoor_adjustment_sets('X', 'Y')

    if not sets:
        raise ValueError('No feasible adjustment set.')

    minimal_adjustment_set = sorted(sets, key=len)[0]
    print(set(sorted(minimal_adjustment_set)))


def main():
    cgm()

if __name__ == '__main__':
    main()
