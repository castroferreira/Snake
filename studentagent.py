# Henrique Manso 65308
# João Gravato 68503
# Andreia Ferreira 65791

from agent import *
from collections import deque
import math


class StudentAgent(Agent):
    def __init__(self, name, body, world):
        super().__init__(name, body, world)
        # Grelha com a mesma dimensão do world
        self.world_grid = self.createWorldGrid(world.size)
        # Tipo de nutriente procurado na jogada anterior.
        self.lastSearchedFoodType = None
        # Tipo de nutriente que quer procurar.
        self.foodTypeNeeded = None 
        # Informação sobre se o outro agente já morreu.
        self.isOtherAgentDead = False 
        # Tipo de nutriente pedido pelo outro agente.
        self.otherAgentFoodTypeNeeded = None
        # Localização de um nutriente enviada pelo outro agente.
        self.otherAgentAdvice = None 
        # Tipo de nutriente enviada pelo outro agente.
        self.otherAgentAdviceType = None 
        # Localização da head do outro agente.
        self.otherAgentPosition = None 

    def chooseAction(self, vision, msg):

        # Verifica se recebou uma mensagem do outro agente.
        # Se já souber que o outro agente morreu, não faz nada.
        if not self.isOtherAgentDead:      
            if msg != b"":
                msg_decoded = msg.decode("utf-8")
                msg_decoded = msg_decoded.replace(" ","")\
                        .replace("(","")\
                        .replace(")","").replace("'","")
                p1,p2,cmd,food = msg_decoded.split(",")

                # Se for um pedido do outro agente
                if cmd == "request": 
                    self.otherAgentPosition = Point(int(p1),int(p2))
                    self.otherAgentFoodTypeNeeded = food

                # Se for uma resposta a um pedido deste agente 
                # ou um aviso de que o outro agente morreu
                if cmd == "response": 
                    if food == 'D':
                        self.isOtherAgentDead = True # O outro agente morreu
                    self.otherAgentAdvice = Point(int(p1),int(p2))
                    self.otherAgentAdviceType = food

        # Se tiver menos de 10 de um dos tipos de nutrientes fica pardo e envia
        # a mensagem 'D', avisando o outro agente de que vai morrer.
        if not self.isOtherAgentDead:
            if (self.nutrients['M'] < 10) or (self.nutrients['S'] < 10):
                req = str((0, 0, "response", "D"))
                return Stay, str.encode(req)

        head = self.body[0]

        # O agent procura o nutriente mais próximo.
        # O tipo de nutriente depende do número de nutrientes que tem.
        # Pode também procurar o nutriente mais próximo independentemente do tipo.
        # Guarda o tipo na váriavel foodTypeNeeded.
        # Se o tipo de nutriente for indiferente, guarda como 'A'.

        cFood = None
        if (self.nutrients['M'] - self.nutrients['S']) > 110:
            cFood = self.closestFood(head, vision, 'S')
            self.foodTypeNeeded = "S"
        elif (self.nutrients['S'] - self.nutrients['M']) > 110:
            cFood = self.closestFood(head, vision, 'M')
            self.foodTypeNeeded = "M"
        else:
            cFood = self.closestFood(head, vision)
            self.foodTypeNeeded = "A"

        # Se não encontrar um nutriente no seu campo de visão mas tiver recebido
        # uma mensagem com a posição de um nutriente, assume essa posição como
        # destino.

        if cFood == None and self.otherAgentAdvice != None:
            cFood = (self.otherAgentAdvice, self.otherAgentAdviceType)

        action = None

        if cFood != None:

            # Se o tipo de nutriente a pesquisar for diferente do que procurou na 
            # jogada anterior, apaga a informação dos nodes visitados.
            if cFood[1] != self.lastSearchedFoodType:
                self.world_grid = self.createWorldGrid(self.world.size)

            # Se o node ainda não tiver sido visitado, registar a primeira visita (com o valor 1)
            # e a distancia de Manhattan do node até ao nutriente
            if self.world_grid[head[0]][head[1]] == 0:
                self.world_grid[head[0]][head[1]] = (1, self.world.dist(head, cFood[0]))
            # Se o node já tiver sido visitado nesta pesquisa, atualizar o número de visitas
            else:
                self.world_grid[head[0]][head[1]] = list(self.world_grid[head[0]][head[1]])
                self.world_grid[head[0]][head[1]][0] += 1
                self.world_grid[head[0]][head[1]] = tuple(self.world_grid[head[0]][head[1]])

            lnewnodes = []

            # get the list of valid actions for us
            validact = ACTIONS[:1]  # staying put is always valid
            for act in ACTIONS[1:]:
                newpos = self.world.translate(head, act)
                if not self.checkIfAlley(newpos, act, vision) \
                        and newpos not in self.world.walls \
                        and newpos not in vision.bodies:
                    validact.append(act)
                    if self.world_grid[newpos[0]][newpos[1]] == 0:
                        self.world_grid[newpos[0]][newpos[1]] = (0, self.world.dist(newpos, cFood[0]))
                    cost = self.world_grid[newpos[0]][newpos[1]][0]
                    heuristic = self.world.dist(newpos, cFood[0])
                    lnewnodes += [Node(newpos, act, cost, heuristic)]

            lnewnodes = sorted(lnewnodes, key=lambda n: n.heuristic + n.cost)

            # Guarda o tipo de comida procurada.
            self.lastSearchedFoodType = cFood[1]

            resp = None

            # Verifica se recebeu um pedido.
            if self.otherAgentFoodTypeNeeded != None:
                advice = self.closestFood(self.otherAgentPosition, vision, self.otherAgentFoodTypeNeeded)
                self.otherAgentFoodTypeNeeded = None
                if advice != None:
                    resp = str((str(advice[0]),"response",advice[1]))

            # Se não existirem novas posições válida, fica parado.
            if lnewnodes == []:              
                if resp != None:
                    return Stay, str.encode(resp)
                return Stay, b""

            # Escolhe a ação da posição com menor custo + heuristica.
            action = lnewnodes[0].act

            if self.otherAgentFoodTypeNeeded != None:
                return action, str.encode(resp)
            return action, b""

        else:
            # Se o tipo de nutriente a pesquisar for diferente de 'A' apaga 
            # a informação dos nodes visitados.
            if self.lastSearchedFoodType != 'A':
                self.world_grid = self.createWorldGrid(self.world.size)

            # Se o node ainda não tiver sido visitado, registar a primeira visita (com o valor 1)
            if self.world_grid[head[0]][head[1]] == 0:
                self.world_grid[head[0]][head[1]] = 1
            # Se o node já tiver sido visitado nesta pesquisa, atualizar o número de visitas
            else:
                self.world_grid[head[0]][head[1]] += 1

            lnewnodes = []

            # get the list of valid actions for us
            validact = ACTIONS[:1]  # staying put is always valid
            for act in ACTIONS[1:]:
                newpos = self.world.translate(head, act)
                if not self.checkIfAlley(newpos, act, vision) \
                        and newpos not in self.world.walls \
                        and newpos not in vision.bodies:
                    validact.append(act)
                    cost = self.world_grid[newpos[0]][newpos[1]]
                    lnewnodes += [Node(newpos, act, cost)]

            lnewnodes = sorted(lnewnodes, key=lambda n: n.cost)

            # Guarda o tipo de nutriente procurado nesta jogada.
            # A -> qualquer tipo.
            self.lastSearchedFoodType = 'A'

            req = b""

            # Verifica se o outro agente está vivo.
            if not self.isOtherAgentDead:
                request = self.foodTypeNeeded
                req = str((str(head), "request", request))

            # Se não existirem novas posições válidas, fica parado.
            if lnewnodes == []:                
                if req != b"":
                    return Stay, str.encode(req)
                return Stay, req

            # Escolhe a ação com menor custo.
            action = lnewnodes[0].act

            # Verifica se o outro agente está vivo.
            if req != b"":
                return action, str.encode(req)
            return action, req

    def closestFood(self, head, vision, foodType=None):
        # Procura na vision deste Agent qual a comida mais proxima
        foods = []
        for key, val in vision.food.items():
            if foodType == 'M' or foodType == 'S':
                if val == foodType:
                    count = 0
                    for i in ACTIONS[1:]:
                        walls = self.world.translate(head, key + i)
                        if walls in self.world.walls and i in vision.bodies:
                            count += 1
                    if count < 3:
                        # tuplo (ponto, tipo de comida, distancia da head até ao ponto)
                        foods += [(key, val, (self.world.dist(key, head)))]

                    elif count == 2:
                        count2 = 0
                        for j in ACTIONS[1:]:
                            test = self.world.translate(i, j)
                            if test in self.world.walls and test in self.world.bodies:
                                count2 += 1
                        if count2 < 3:
                            foods += [(key, val, (self.world.dist(key, head)))]
            else:
                foods += [(key, val, (self.world.dist(key, head)))]

        # se nao existir food do tipo foodType na vision, retorna None
        if foods == []:
            return None

        # ordena a lista de tuplos pela distancia
        foods = sorted(foods, key=lambda x: x[2])
        # retorna o ponto (apenas as coordenadas) com menor distancia
        return (foods[0][0], foods[0][1])

    def actionForClosestFood(self, validact, food, vision, head):
        actions = []
        for a in validact:
            newpos = self.world.translate(head, a)
            if newpos not in self.world.walls and newpos not in vision.bodies:
                for b in validact:
                    auxpos = self.world.translate(head, a + b)
                    if auxpos not in self.world.walls and auxpos not in vision.bodies:
                        # tuplo (action, distancia entre nova posiçao e a comida)
                        actions += [(a, self.world.dist(newpos, food))]
        if len(actions) > 0:
            # ordena a lista de tuplos pela distancia
            actions = sorted(actions, key=lambda x: x[1])
            # retorna a action com menor distancia
            return actions[0][0]
        return random.choice(validact)

    def createWorldGrid(self, size):
        (h, v) = size
        grid = [[0 for j in range(v)] for n in range(h)]
        return grid

    def checkIfAlley(self, newpos, act, vision):

        # Verificação de becos com profundidade até 2

        aux_up = Left
        aux_down = Right
        aux_left = Up
        aux_right = Down

        act_up = self.world.translate(newpos, aux_up)
        act_down = self.world.translate(newpos, aux_down)
        act_left = self.world.translate(newpos, aux_left)
        act_right = self.world.translate(newpos, aux_right)

        act_up_up = self.world.translate(act_up, aux_up)
        act_left_up = self.world.translate(act_left, aux_up)
        act_right_up = self.world.translate(act_right, aux_up)

        act_down_down = self.world.translate(act_down, aux_down)
        act_left_down = self.world.translate(act_left, aux_down)
        act_right_down = self.world.translate(act_right, aux_down)

        act_left_left = self.world.translate(act_left, aux_left)
        act_up_left = self.world.translate(act_up, aux_left)
        act_down_left = self.world.translate(act_down, aux_left)

        act_right_right = self.world.translate(act_right, aux_right)
        act_up_right = self.world.translate(act_up, aux_right)
        act_down_right = self.world.translate(act_down, aux_right)

        if act == aux_up:
            if newpos not in self.world.walls and (act_up in self.world.walls or act_up in vision.bodies) and \
                    (act_left in self.world.walls or act_left in vision.bodies) and \
                    (act_right in self.world.walls or act_right in vision.bodies):
                return True
            if (newpos not in self.world.walls) and (act_up not in self.world.walls) and (
                            act_left in self.world.walls or act_left in vision.bodies) and \
                    (act_right in self.world.walls or act_right in vision.bodies) and \
                    (act_up_up in self.world.walls or act_up_up in vision.bodies) and \
                    (act_left_up in self.world.walls or act_left_up in vision.bodies) and \
                    (act_right_up in self.world.walls or act_right_up in vision.bodies):
                return True

        if act == aux_left:
            if newpos not in self.world.walls and (act_up in self.world.walls or act_up in vision.bodies) and \
                    (act_left in self.world.walls or act_left in vision.bodies) and \
                    (act_down in self.world.walls or act_down in vision.bodies):
                return True
            if (newpos not in self.world.walls) and (act_left not in self.world.walls) and (
                            act_up in self.world.walls or act_up in vision.bodies) and \
                    (act_down in self.world.walls or act_down in vision.bodies) and \
                    (act_left_left in self.world.walls or act_left_left in vision.bodies) and \
                    (act_up_left in self.world.walls or act_up_left in vision.bodies) and \
                    (act_down_left in self.world.walls or act_down_left in vision.bodies):
                return True

        if act == aux_right:
            if newpos not in self.world.walls and (act_up in self.world.walls or act_up in vision.bodies) and \
                    (act_right in self.world.walls or act_right in vision.bodies) and \
                    (act_down in self.world.walls or act_down in vision.bodies):
                return True
            if (newpos not in self.world.walls) and (act_right not in self.world.walls) and (
                            act_up in self.world.walls or act_up in vision.bodies) and \
                    (act_down in self.world.walls or act_down in vision.bodies) and \
                    (act_right_right in self.world.walls or act_right_right in vision.bodies) and \
                    (act_up_right in self.world.walls or act_up_right in vision.bodies) and \
                    (act_down_right in self.world.walls or act_down_right in vision.bodies):
                return True

        if act == aux_down:
            if newpos not in self.world.walls and (act_down in self.world.walls or act_down in vision.bodies) and \
                    (act_left in self.world.walls or act_left in vision.bodies) and \
                    (act_right in self.world.walls or act_right in vision.bodies):
                return True
            if (newpos not in self.world.walls) and (act_down not in self.world.walls) and (
                            act_left in self.world.walls or act_left in vision.bodies) and \
                    (act_right in self.world.walls or act_right in vision.bodies) and \
                    (act_down_down in self.world.walls or act_down_down in vision.bodies) and \
                    (act_left_down in self.world.walls or act_left_down in vision.bodies) and \
                    (act_right_down in self.world.walls or act_right_down in vision.bodies):
                return True

        else:
            return False

class Node:
    def __init__(self, pos, act, cost=None, heuristic=None):
        self.pos = pos
        self.heuristic = heuristic
        self.cost = cost
        self.act = act
        self.numvis = 0