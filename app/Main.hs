{-# LANGUAGE FlexibleContexts #-}
module Main where

import           System.Exit
import           System.Random
import           System.TimeIt
import           Control.Monad.State
import           Numeric.LinearAlgebra
import           Control.Monad.Trans.Maybe      ( runMaybeT )
import           Control.Monad.Writer           ( runWriter )
import           Criterion.Main
import           Criterion.Types

import           Neural.Network
import           Neural.Activation              ( sigmoid
                                                , sigmoid'
                                                )
import           Neural.Training
import           Data.MNIST                     ( loadData )

shallowShape :: [Int]
shallowShape = [784, 30, 10]

deepShape :: [Int]
deepShape = [784, 30, 30, 30, 10]

-- HYPER PARAMETERS
-- learing rate
μ :: Double
μ = 0.5

-- Mini batch size
mbs :: Int
mbs = 100

-- Amount of training epochs
epochs :: Int
epochs = 50

main :: IO ()
main = defaultMain
  [ env setupEnv $ \ ~(trainEx, testEx) ->
      let trainShallow = timeIt (randomIO >>= trainNetwork trainEx testEx shallowShape)
          trainDeep    = timeIt (randomIO >>= trainNetwork trainEx testEx deepShape)
      in  bgroup
            "main"
            [ bench "shallow" $ nfIO trainShallow
            , bench "deep"    $ nfIO trainDeep
            ]
  ]

trainNetwork
  :: [TrainingExample Double]
  -> [TestingExample Double]
  -> [Int]
  -> Seed
  -> IO (Network Double)
trainNetwork trainEx testEx shape seed = do
  let gen             = mkStdGen seed
  let trainFunc = train sigmoid sigmoid' μ trainEx (Just testEx) epochs mbs
  let (randNet, gen') = runState (randomNetwork shape) gen
  let (net, logs) = runWriter $ evalStateT (trainFunc randNet) gen'
  putStrLn "Training network..."

  mapM_ putStrLn logs
  return net

setupEnv
    :: IO ([TrainingExample Double], [TestingExample Double])
setupEnv = do
  mnistData <- runMaybeT loadData
  case mnistData of
    Nothing -> putStrLn "Failed to load MNIST data." >> exitFailure
    Just (trainEx, testEx) -> return (trainEx, testEx)
