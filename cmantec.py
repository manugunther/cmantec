#!/usr/bin/python
# -*- coding: UTF-8 -*-

from random import uniform,randint,shuffle
from math import exp,sqrt
import pdb


def maxindex(l):
    r = 0
    for i in range(len(l)):
        if l[i] > l[r]:
            r = i
    
    return r

class CMNet(object):
    
    
    def __init__(self,nunits,t0,imax,gfac,npats,patterns,targets,filterNoise=True,nsig=2.5):
        
        
        self.ninput = nunits + 1
      
        # Input layer. La inicializamos a 0.
        self.input = [0 for i in range(self.ninput)]
        
        # Hidden layer. Inicialmente tiene una sola neurona y la inicializamos a 0
        self.nhidden = 1
        self.hidden = [0]
        # Número de iteraciones para cada neurona de hidden layer.
        self.itershidden = [0]
        # Temperatura de cada neurona de hidden layer.
        self.temphidden = [t0]
        
        # Output layer. Tiene una sola neurona.
        self.output = [0]
        
        # Pesos entre input layer y hidden layer.
        self.w = [[0.0 for i in range(self.ninput)]]
        
        # Entre hidden y output los pesos son iguales a 1.
        
        
        # Temperatura inicial
        self.t0 = nunits
        
        # cantidad máxima de iteraciones
        self.imax = imax
        
        # Constante gfac
        self.gfac = gfac
        
        # Cantidad de patrones de aprendizaje
        self.npats = npats
        
        # Patrones de aprendizaje con el resultado esperado
        self.patterns = list(patterns)
        self.targets = list(targets)
        
        # Número de patrón presentado
        self.presentedPattern = -1
        
        
        # Factor de descenso de temperatura
        self.descTemp = float(self.t0) / float(self.imax)
        
        # Corrección de ruido en los patrones
        self.filterNoise = filterNoise
        
        self.nsig = nsig
        
        
        # Cantidad de veces que se presenta cada patrón en UN CICLO DE APRENDIZAJE
        self.nPres = [0 for i in range(npats)]
        # Cantidad de veces q se clasifica mal cada patrón en UN CICLO DE APRENDIZAJE
        self.badClassif = [0 for i in range(npats)]

        self.cmantec()
        
        
        
    def presentPattern(self,u):
        self.input = list(self.patterns[u])
        self.input.append(1.0) # bias
        
        self.presentedPattern = u
        
    # Función de activación de la neurona de salida.
    # Los pesos sinápticos son iguales a 1, de manera que computa la majority function.
    def activationOutput(self):
        
        s = 0.0
        for i in self.hidden:
            s = s + i
            
        threshold = (self.nhidden / 2.0) - 0.5
        
        res = 0
        
        if s - threshold >= 0:
            res = 1
            
        return res
        
    
    # Función de activación de la neurona j en la capa oculta
    def activationHidden(self,j):
        
        fi = 0.0
        for i in range(self.ninput):
            fi = fi + self.w[j][i] * self.input[i]

        res = 0
        
        
        if fi >= 0:
            res = 1
        
        return (fi,res)
        
        
        
    def calculateTFac(self,j,fi):
        
        t1 = (self.temphidden[j]/self.t0)
        t2 = exp(- abs(fi)/self.temphidden[j])
        
        return t1*t2
        
        
    def deltaWeight(self,i,s,tfac):
        assert(self.presentedPattern >= 0)
        
        u = self.presentedPattern
        
        return (self.targets[u] - s) * self.input[i] * tfac
    
    # Realiza un cálculo. El valor de entrada debe tener self.ninput bits.
    # Al llamar a esta función, la red no tiene presentado ningún patrón.
    def calculate(self,value):
        self.presentedPattern = None
        self.input = list(value)
        self.input.append(1.0) # bias
        
        for j in range(self.nhidden):
            (fi,v) = self.activationHidden(j)
            self.hidden[j] = v
            
        return self.activationOutput()
        
    def calculatePattern(self):
        for j in range(self.nhidden):
            (fi,v) = self.activationHidden(j)
            self.hidden[j] = v
            
        return self.activationOutput()
        
    def modifyWeights(self,j,s,tfac):
        for i in range(self.ninput):
            delta = self.deltaWeight(i,s,tfac)
            self.w[j][i] += delta

        self.itershidden[j] += 1
        self.temphidden[j] -= self.descTemp
        
    def addNeuron(self):
        self.hidden.append(0)
        self.nhidden += 1
        self.itershidden.append(0)
        self.temphidden.append(self.t0)
        
        self.w.append([0.0 for i in range(self.ninput)])
        
    def restartTemperatures(self):
        for i in range(self.nhidden):
            self.itershidden[i] = 0
            self.temphidden[i] = self.t0
    
    def cmantec(self):
        
        patterns_ok = [False for i in range(self.npats)]
        
        allpatterns_ok = False
      
        u = 0
        while(not allpatterns_ok):      
        
            u = randint(0,self.npats-1)
            
            self.presentPattern(u)
            
            self.nPres[u] += 1
    
            value = self.calculatePattern()
           
            
            if value!=self.targets[u]:
                #print "Patrón mal clasificado"
                
                # Se clasificó mal el patrón
                self.badClassif[u] += 1
                
                patterns_ok = [False for i in range(self.npats)]
                
                tfacs = []
                for j in range(self.nhidden):
                    (fi,_) = self.activationHidden(j)
                    tfacs.append(self.calculateTFac(j,fi))
                    
                nmaxtfac = maxindex(tfacs)
                
                
                if tfacs[nmaxtfac] > self.gfac:
                    self.modifyWeights(nmaxtfac,self.hidden[nmaxtfac],tfacs[nmaxtfac])
                    
                else:
                    # TERMINO CICLO DE APRENDIZAJE
                    self.addNeuron()
                    self.restartTemperatures()
                    
                    if self.filterNoise:
                        self.removePatterns()
                        self.nPres = [0 for i in range(self.npats)]
                        self.badClassif = [0 for i in range(self.npats)]
                    
                    patterns_ok = [False for i in range(self.npats)]
                    
                    print "nhidden = ",self.nhidden
                    
            else:
                patterns_ok[u] = True
                
                allpatterns_ok = True
                for i in range(self.npats):
                    allpatterns_ok = allpatterns_ok and patterns_ok[i]
                
                
    # Testeamos que todos los patterns son clasificados correctamente
    def test(self):
        l = []
        
        res = True
        for u in range(self.npats):
            self.presentPattern(u)
            value = self.calculatePattern()
            
            res = res and (value == self.targets[u])
            
        
        return res
            
    def removePatterns(self):
        
        # lista de la cantidad de veces que los patrones fueron mal clasificados
        badPres = [self.badClassif[i] for i in range(self.npats) if self.nPres[i] > 0]
        npresents = len(badPres)
        
        # Calculamos la media de mal clasificación de patrones
        meanPres = sum(badPres) / float(npresents)
        desvPres = sum([pow(j - meanPres,2) for j in badPres]) / npresents
        desvPres = sqrt(desvPres)
        
        pattToElim = [(self.patterns[u],self.targets[u],u) for u in range(self.npats) if self.nPres[u] > 0 and self.badClassif[u] - meanPres > self.nsig * desvPres]
        
        print "Media de malas clasificaciones = ",meanPres
        
        for (pat,targ,u) in pattToElim:
            print "Elimino patrón", pat
            print "nPres[u] = ",self.nPres[u]
            print "badClass[u] = ",self.badClassif[u]
            
            self.patterns.remove(pat)
            self.targets.remove(targ)
            
            self.npats -= 1
            print "npats = ",self.npats    

    
def example(fileinput,ntraining,normalize=True,noiseFilter=True):
    
    f = open(fileinput,'r')
    
    data = []
    
    for line in f:
        
        l = line.split()
        l = map(float,l)
        
        data.append(list(l))
    
    nattribs = len(data[0]) - 1
    
    shuffle(data)
    
    # Tomamos 100 ejemplos para entrenar
    
    l = data[0:ntraining]
    patterns = map(lambda t: t[0:nattribs],l)
    
    targets = map(lambda t: t[nattribs], l)
    
    npat = len(patterns)
    
    if normalize:
        patterns = normalizePatterns(patterns,nattribs)
    
    #print "patterns normalizados = ",patterns
    
    n = CMNet(nattribs,nattribs,20000,0.05,npat,patterns,targets,noiseFilter)
    
    # Nos aseguramos que se hayan aprendido todos los patrones
    assert(n.test())
    
    # Ahora vemos la capacidad de generalizacion de la red
    l = data[ntraining:len(data)]
    inputs = map(lambda t: t[0:nattribs],l)
    results = map(lambda t: t[nattribs], l)
        
    
    if normalize:
        inputs = normalizePatterns(inputs,nattribs)
    
    nrights = 0
    
    for i in range(len(inputs)):
        out = n.calculate(inputs[i])
        
        if out == results[i]:
            nrights += 1

    genCapacity = nrights / float(len(inputs))
        
    print "Aprendizaje con ",npat ," ejemplos. Generalizo con ",len(inputs),"ejemplos. Capacidad de Generalización = ",genCapacity
    
    f.close()
    
    
def normalizePatterns(ldata,nattribs):
    
    
    ltrans = transposeList(ldata,nattribs)
    ltn = []
    
    for l in ltrans:
        
        ltn.append(normalizeList(l))
    
    return transposeList(ltn,len(ldata))

    
def normalizeList(l):
    maxI = float(max(l))
    minI = float(min(l))
    
    fNorm = lambda x: (x - minI)/(maxI - minI)
    
    return map(fNorm,l)
    
# Transpone una matriz de len(l) x n
def transposeList(l,n):
    
    r = []
    m = len(l)
    
    for i in range(n):
        t = []
        for j in range(m):
            t.append(l[j][i])

        r.append(list(t))
    
    return r
    



