module Main where

import           System.Random
import           Control.Monad.State
import Numeric.LinearAlgebra

import           Neural.Network
import Neural.Activation (sigmoid)

seed :: Int
seed = 114117116104

main :: IO ()
main = do
  let gen     = mkStdGen seed
  let network = evalState (newNetwork [3, 5, 8, 2]) gen :: Network Double
  let input   = fromList [10, -4, 0] :: Vector Double
  let ff      = feedforwards sigmoid input network
  putStrLn "Feedforward output: "
  print ff
