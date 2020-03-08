{-# LANGUAGE FlexibleContexts #-}
module Main where

import           System.Random
import           Control.Monad.State
import           Numeric.LinearAlgebra
import           Control.Monad.Trans.Maybe      ( runMaybeT )
import           Control.Monad.Writer           ( runWriter )

import           Neural.Network
import           Neural.Activation              ( sigmoid
                                                , sigmoid'
                                                )
import           Neural.Training
import           Data.MNIST                     ( loadData )

seed :: Int
seed = 114117116106

layerSizes :: [Int]
layerSizes = [784, 40, 30, 10]

main :: IO ()
main = do
  mnistData <- runMaybeT loadData
  case mnistData of
    Nothing                -> putStrLn "Failed to load MNIST data."
    Just (trainEx, testEx) -> void $ trainNetwork trainEx testEx

trainNetwork
  :: (Numeric a, Floating a, Element a, Random a, Num (Vector a))
  => [TrainingExample a]
  -> [TestingExample a]
  -> IO (Network a)
trainNetwork trainEx testEx = do
  let gen             = mkStdGen seed
  let trainFunc = train sigmoid sigmoid' 3.0 trainEx Nothing 30 100
  let (randNet, gen') = runState (randomNetwork layerSizes) gen
  let (net, logs) = runWriter $ evalStateT (trainFunc randNet) gen'
  mapM_ putStrLn logs
  return net
