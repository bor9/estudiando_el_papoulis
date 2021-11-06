from scipy.optimize import fmin

def shortest_confidence_interval(dist_name, confidence_coef=0.95, **args):
    """ Cálculo del intervalo de confianza mas corto.

    :param dist_name: nombre scipy de la distribución (str)
    :param confidence_coef: coeficiente de confianza (float entre 0 y 1)
    :param args: argumentos scipy de la distribución
    :return: límite inferior y superior del intervalo (array [low, high])

    Ejemplo de uso:

    c1, c2 = shortest_confidence_interval(beta, confidence_coef=0.9, a=5, b=3)
    """
    # configuración de la distribución con los argumentos dados
    distri = dist_name(**args)
    # estimación inicial de la probabilidad (área) de la cola inferior
    low_tail_pr_ini = 1.0 - confidence_coef

    def interval_width(low_tail_pr):
        """
        Cálculo del intervalo de confianza cuando el área de la cola inferior es
        low_tail_pr, por lo que el límite inferior es F^{-1}(low_tail_pr).

        :param low_tail_pr: probabilidad de la cola inferior (float entre 0 y 1)
        :return: largo del intervalo (float)
        """
        return distri.ppf(confidence_coef + low_tail_pr) - distri.ppf(low_tail_pr)

    # Busqueda de la probabilidad de la cola inferior (low_tail_pr) que minimiza el
    # intervalo de confianza (función interval_width) fijando el coeficiente de confianza.
    low_tail_pr_short = fmin(interval_width, low_tail_pr_ini, ftol=1e-8, disp=False)[0]
    # Se retorna el intervalo como un array([low, high])
    return distri.ppf([low_tail_pr_short, confidence_coef + low_tail_pr_short])