module Main where

import           System.Random
import           Control.Monad.State

import           Neural.Network

seed :: Int
seed = 114117116104

main :: IO ()
main = do
  let gen     = mkStdGen seed
  let network = evalState (newNetwork [3, 5, 8, 2]) gen :: Network Double
  print network
