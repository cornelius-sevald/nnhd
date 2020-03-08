{-# LANGUAGE FlexibleContexts #-}
module Neural.Training
  ( TrainingExample(..)
  , TestingExample(..)
  , train
  , backpropagation
  )
where

import           Foreign.Marshal.Utils          ( fromBool )
import           Control.Monad
import           Control.Monad.Writer
import           Control.Monad.State
import           Control.Monad.Random
import           System.Random
import           System.Random.Shuffle          ( shuffleM )
import           Numeric.LinearAlgebra
import           Text.Printf

import           Debug.Trace

import           Neural.Activation
import qualified Neural.Network                as N
import           Neural.Network          hiding ( weights
                                                , biases
                                                )
import           Neural.Layer                   ( Layer(..) )

-- | A training example with an input and desired output in vector form.
data TrainingExample a = TrainingExample (Vector a) (Vector a) deriving Show
-- | A training example with an input and desired output as an index.
data TestingExample a  = TestingExample (Vector a) Int deriving Show

-- | The Hadamar product, elementwise multiplication.
(⊙) :: Num (Vector a) => Vector a -> Vector a -> Vector a
u ⊙ v = u * v


-- | Train a neural network over some amount of epochs given some training
-- data and the size of the mini batches.
train
  :: (RandomGen g, Fractional a, Numeric a, Num (Vector a))
  => ActivationFunction a      -- ^ The activation function
  -> ActivationFunction' a     -- ^ The derivative of the activation function
  -> a                         -- ^ The learning rate
  -> [TrainingExample a]       -- ^ Training data
  -> Maybe [TestingExample a]  -- ^ Optional testing data
  -> Int                       -- ^ Number of ephocs
  -> Int                       -- ^ Mini batch size
  -> Network a                 -- ^ The untrained network
  -> StateT g (Writer [String]) (Network a) -- ^ The trained network
train σ σ' η trainingData testData epochs mbs net = do
  let tellProgress epoch net' = case testData of
        Nothing        -> tell ["Epoch " ++ show epoch ++ " complete"]
        Just testData' -> tell
          [printf "Epoch %d: %d / %d (%.2f%%)" epoch evaluation n p]
         where
          evaluation = evaluate σ testData' net'
          n          = length testData'
          p          = fromIntegral evaluation / fromIntegral n * 100 :: Double
  let trainEpoch net' epoch = do
        batches <- state $ runRand $ forM
          [1 .. epochs]
          (\_ -> take mbs <$> shuffleM trainingData)
        let tmb        = trainMiniBatch σ σ' η
        let trainedNet = foldl (flip tmb) net' batches
        lift $ tellProgress epoch trainedNet
        return trainedNet

  foldM trainEpoch net [1 .. epochs]

-- | Evaluate the network assuming the output is the index of the neuron with
-- the highest activation.
evaluate
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a     -- ^ The activation function
  -> [TestingExample a]       -- ^ Testing data
  -> Network a                -- ^ The untrained network
  -> Int                      -- ^ The amount of tests the network passed
evaluate σ testData net =
  let eval (TestingExample x y) = maxIndex (fst $ feedforward σ x net) == y
      results = map eval testData
  in  foldl (\acc r -> acc + fromBool r) 0 results

-- | Train a neural network for a mini batch of training data.
trainMiniBatch
  :: (Fractional a, Numeric a, Num (Vector a))
  => ActivationFunction a    -- ^ The activation function
  -> ActivationFunction' a   -- ^ The derivative of the activation function
  -> a                       -- ^ The learning rate
  -> [TrainingExample a]     -- ^ A mini batch of training examples
  -> Network a               -- ^ The untrained network
  -> Network a               -- ^ The trained network
trainMiniBatch σ σ' η batch net =
  let bp example = backpropagation σ σ' example net
      η'     = η / fromIntegral (length batch)
      ws0    = map (konst 0 . size) (N.weights net)
      bs0    = map (konst 0 . size) (N.biases net)
      deltas = map (unzip . bp) batch
      f (w, b) (dw, db) = (zipWith (+) w dw, zipWith (+) b db)
      (nabla_w, nabla_b) = foldl f (ws0, bs0) deltas
      ws = zipWith (\w nw -> w - scale η' nw) (N.weights net) nabla_w
      bs = zipWith (\b nb -> b - scale η' nb) (N.biases net) nabla_b
  in  newNetwork ws bs

-- | Find the gradient of the cost function with respects to a networks
-- weights and biases for one training example.
backpropagation
  :: (Numeric a, Num (Vector a))
  => ActivationFunction a    -- ^ The activation function
  -> ActivationFunction' a   -- ^ The derivative of the activation function
  -> TrainingExample a       -- ^ A single training example
  -> Network a               -- ^ The neural network
  -> [(Matrix a, Vector a)]  -- ^ The weight and bias gradient of the
                               -- quadratic cost function
backpropagation σ σ' (TrainingExample x y) net@(Network layers) =
  let (as, zs) = unzip $ feedforwards σ x net   -- Feed forward and save the
                                                -- results
      as_      = x : as
      nabla_aC = last as - y                    -- The gradient of the cost
                                                -- function
      calc_δ (l, z) δ = bp2 σ' (weights l) δ z  -- A function that calculates
                                                -- the error for the previous
                                                -- layer
      δ0      = bp1 σ' nabla_aC (last zs)       -- The error of the last layer
      δs      = scanr calc_δ δ0 (zip (tail layers) zs)
      nabla_b = δs                              -- The 3rd equation
      nabla_w = zipWith outer δs as_            -- The 4th equation
  in  zip nabla_w nabla_b

-- | The first equation of backpropagation.
bp1
  :: (Numeric a, Num (Vector a))
  => ActivationFunction' a    -- ^ The derivative of the activation function
  -> Vector a                 -- ^ The gradient of the cost function
                                -- with respects to the activation
  -> Vector a                 -- ^ The weighted input
  -> Vector a                 -- ^ The error in the output layer
bp1 _σ' nabla_aC z = nabla_aC ⊙ σ' z where σ' = cmap _σ'

-- | The second equation of backpropagation.
bp2
  :: (Numeric a, Num (Vector a))
  => ActivationFunction' a    -- ^ The derivative of the activation function
  -> Matrix a                 -- ^ The weights of the next layer
  -> Vector a                 -- ^ The error of the next layer
  -> Vector a                 -- ^ The weighted input of the current layer
  -> Vector a                 -- ^ The error of the current layer
bp2 _σ' w δ z = (tr' w #> δ) ⊙ σ' z where σ' = cmap _σ'

