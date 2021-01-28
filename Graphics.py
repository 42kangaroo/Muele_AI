import PySimpleGUI as psGui
import numpy as np

import Network
import configs
from MillEnv import MillEnv
from mcts import State, MonteCarloTreeSearch, generate_empty_nodes


class MillDisplayer(object):
    def __init__(self, MillEnvironment: MillEnv = None):
        psGui.theme("dark")
        self.millImage: str = "MühleBrett.png"
        self.blackCheckerImage: str = "Schwarz.png"
        self.whiteCheckerImage: str = "Weiss.png"
        self.millEnv: MillEnv = MillEnv()
        self.moves_names = {0: "put down", 1: "choose checker", 2: "move selected checker",
                            3: "delete opponent checker"}
        self.last_move = []
        if MillEnvironment is not None:
            self.millEnv = MillEnvironment
        self.ImageIDArray = np.array([])
        self.imageLocations = [(10, 490), (225, 490), (440, 490),
                               (75, 415), (225, 415), (375, 415),
                               (150, 340), (225, 340), (310, 340),
                               (10, 265), (75, 265), (150, 265),
                               (310, 265), (375, 265), (440, 265),
                               (150, 190), (225, 190), (310, 190),
                               (75, 115), (225, 115), (375, 115),
                               (10, 55), (225, 55), (440, 55)]
        self.graph = psGui.Graph(
            canvas_size=(500, 500),
            graph_bottom_left=(0, 0),
            graph_top_right=(500, 500),
        )
        self.statusTextBox = psGui.Text("Player " + self.getPlayerName(self.millEnv.isPlaying) + " is playing",
                                        size=(60, 1))
        self.layout_ = [
            [psGui.Button("Player vs. Player"), psGui.Button("Player vs. Agent"), psGui.Button("Agent vs. Agent")],
            [self.statusTextBox],
            [self.graph],
            [psGui.Button("Close")]]
        self.window = psGui.Window("Mill AI", layout=self.layout_, finalize=True)
        self.window.finalize()
        self.graph.DrawImage(filename=self.millImage, location=(0, 500))
        self.activateClick()
        self.reloadEnv()

    def windowsLoop(self):
        while True:
            event, values = self.window.read()
            if event == psGui.WIN_CLOSED or event == 'Close':  # if user closes window or clicks cancel
                break
            elif not event == "":
                self.reset()
        self.window.close()

    def makeMove(self, pos: int) -> bool:
        valid, reward = self.millEnv.makeMove(pos)
        if valid:
            self.reloadEnv()
        return valid

    def reloadEnv(self):
        self.setStatus(
            "Player " + self.getPlayerName(self.millEnv.isPlaying) + " is playing - move needed: " + self.moves_names[
                self.millEnv.moveNeeded])
        for imageID in self.ImageIDArray:
            self.graph.DeleteFigure(imageID)
        self.ImageIDArray = np.array([])
        for case, location in zip(self.millEnv.getBoard(), self.imageLocations):
            if case == 1:
                self.ImageIDArray = np.append(self.ImageIDArray,
                                              self.graph.DrawImage(filename=self.blackCheckerImage, location=location))
            elif case == -1:
                self.ImageIDArray = np.append(self.ImageIDArray,
                                              self.graph.DrawImage(filename=self.whiteCheckerImage, location=location))
        self.window.refresh()

    def getClicked(self, event) -> int:
        for index, location in enumerate(self.imageLocations):
            x2, y2 = location
            if self.isInArea(event.x, -event.y + 500, x2, y2, 50, 50):
                return index
        return -1

    def setAfterClicked(self, event):
        pos = self.getClicked(event)
        if pos == -1:
            return
        if self.millEnv.moveNeeded == 2:
            dif = self.millEnv.selected - pos
            if dif == 0:
                return
            if dif == -1:
                pos = 1
            elif dif == 1:
                pos = 3
            elif dif < 0:
                pos = 2
            elif dif > 0:
                pos = 0
        self.last_move.append(pos)
        if not self.makeMove(pos):
            self.last_move.pop()

    def isInArea(self, posX1: int, posY1: int, posX2: int, posY2: int, width: int, height: int) -> bool:
        if posX2 <= posX1 <= posX2 + width:
            if posY2 >= posY1 >= posY2 - height:
                return True
        return False

    def setStatus(self, status: str):
        self.statusTextBox.Update(status)

    def close(self):
        self.window.close()

    def activateClick(self):
        self.graph.TKCanvas.bind("<Button-1>", self.setAfterClicked)

    def deactivateClick(self):
        self.graph.TKCanvas.unbind("<Button-1>")

    def read(self, timout: bool = False):
        return self.window.read(0 if timout else None)

    def reset(self):
        self.millEnv.reset()
        self.reloadEnv()

    def getPlayerName(self, player: int) -> str:
        if player == 1:
            return "black"
        elif player == -1:
            return "white"
        else:
            return "not a player"


class ModeratedGraphics(object):
    def __init__(self, network_path, faktor, exponent):
        self.nnet = Network.get_net(configs.FILTERS, configs.KERNEL_SIZE, configs.HIDDEN_SIZE, configs.OUT_FILTERS,
                                    configs.OUT_KERNEL_SIZE, configs.NUM_ACTIONS, configs.INPUT_SIZE)
        self.nnet.load_weights(network_path)
        self.exponent = exponent
        self.faktor = faktor
        self.env = MillEnv()
        self.graphics = MillDisplayer(self.env)
        self.graphics.reloadEnv()
        self.root: State = State(np.zeros((1, 24)), 0, -self.env.isPlaying, self.env)
        val = self.root.setValAndPriors(self.nnet)
        self.root.backpropagate(val)
        self.mcts = MonteCarloTreeSearch(self.root)

    def agentPlay(self):
        self.resetMonteCarlo()
        self.graphics.deactivateClick()
        finished = 0
        while finished == 0:
            self.graphics.reloadEnv()
            pi = self.mcts.search(self.nnet, self.faktor, self.exponent)
            if self.mcts.depth < 5:
                choices_pi = np.where(pi == -1, np.zeros(pi.shape), pi)
                pos = np.random.choice(np.arange(24), p=choices_pi)
            else:
                pos = np.argmax(pi)
            self.mcts.goToMoveNode(pos)
            self.env.setFullState(self.mcts.root.state[0], self.mcts.root.state[1], self.mcts.root.state[2],
                                  self.mcts.root.state[3], self.mcts.root.state[4], self.mcts.root.state[5],
                                  self.mcts.root.state[6], self.mcts.root.state[7], self.mcts.root.state[8],
                                  self.mcts.root.state[9])
            event, values = self.graphics.read(True)
            if self.eventHandler(event):
                return
            finished = self.env.isFinished()
        self.graphics.reloadEnv()
        if not finished == 2:
            self.graphics.setStatus("player " + self.graphics.getPlayerName(finished) + " won")
        else:
            self.graphics.setStatus("The game ended in a draw")

    def playersVSPlayer(self):
        self.graphics.activateClick()
        self.graphics.reset()
        finished = 0
        while finished == 0:
            event, values = self.graphics.read()
            if self.eventHandler(event):
                return
            self.graphics.reloadEnv()
            finished = self.env.isFinished()
        self.graphics.reloadEnv()
        if not finished == 2:
            self.graphics.setStatus("player " + self.graphics.getPlayerName(finished) + " won")
        else:
            self.graphics.setStatus("The game ended in a draw")
        self.graphics.deactivateClick()

    def playerVSAgent(self):
        self.graphics.activateClick()
        self.resetMonteCarlo()
        finished = 0
        while finished == 0:
            event, values = self.graphics.read(True)
            if self.eventHandler(event):
                return
            while len(self.graphics.last_move) > 0:
                self.mcts.goToMoveNode(self.graphics.last_move.pop())
                self.env.setFullState(self.mcts.root.state[0], self.mcts.root.state[1], self.mcts.root.state[2],
                                      self.mcts.root.state[3], self.mcts.root.state[4], self.mcts.root.state[5],
                                      self.mcts.root.state[6], self.mcts.root.state[7], self.mcts.root.state[8],
                                      self.mcts.root.state[9])
            if self.env.isPlaying == 1:
                self.graphics.activateClick()
            else:
                if self.mcts.root.priors is None:
                    val = self.mcts.root.setValAndPriors(self.nnet)
                    self.mcts.root.backpropagate(val)
                    generate_empty_nodes(self.mcts.root)
                self.graphics.deactivateClick()
                pi = self.mcts.search(self.nnet, self.faktor, self.exponent)
                pos = np.argmax(pi)
                self.mcts.goToMoveNode(pos)
                self.env.setFullState(self.mcts.root.state[0], self.mcts.root.state[1], self.mcts.root.state[2],
                                      self.mcts.root.state[3], self.mcts.root.state[4], self.mcts.root.state[5],
                                      self.mcts.root.state[6], self.mcts.root.state[7], self.mcts.root.state[8],
                                      self.mcts.root.state[9])
            self.graphics.reloadEnv()
            finished = self.env.isFinished()
        self.graphics.reloadEnv()
        if not finished == 2:
            self.graphics.setStatus("player " + self.graphics.getPlayerName(finished) + " won")
        else:
            self.graphics.setStatus("The game ended in a draw")
        self.graphics.deactivateClick()

    def playLoop(self):
        self.graphics.deactivateClick()
        self.playerVSAgent()
        finished = False
        while not finished:
            event, values = self.graphics.read()
            finished = self.eventHandler(event)
            if event != psGui.WIN_CLOSED and event != 'Close':
                finished = False

    def eventHandler(self, event) -> bool:
        if event == psGui.WIN_CLOSED or event == 'Close':  # if user closes window or clicks cancel
            self.graphics.close()
            return True
        elif event == "Agent vs. Agent":
            self.agentPlay()
            return True
        elif event == "Player vs. Player":
            self.playersVSPlayer()
            return True
        elif event == "Player vs. Agent":
            self.playerVSAgent()
            return True
        return False

    def resetMonteCarlo(self):
        self.env.reset()
        self.graphics.millEnv.reset()
        self.root: State = State(np.zeros((1, 24)), 0, -self.env.isPlaying, self.env)
        self.env.setFullState(self.mcts.root.state[0], self.mcts.root.state[1], self.mcts.root.state[2],
                              self.mcts.root.state[3], self.mcts.root.state[4], self.mcts.root.state[5],
                              self.mcts.root.state[6], self.mcts.root.state[7], self.mcts.root.state[8],
                              self.mcts.root.state[9])
        self.mcts = MonteCarloTreeSearch(self.root)
        val = self.mcts.root.setValAndPriors(self.nnet)
        self.mcts.root.backpropagate(val)
        self.graphics.reloadEnv()


if __name__ == "__main__":
    MCGraphics = ModeratedGraphics("models/episode-2.h5", 3, 1.15)
    MCGraphics.playLoop()
