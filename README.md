# About Pull Request

This repository is read-only, so Pull Request is not accepted. Thank you for your understanding.

# Ariadne

Ariadne is a shogi engine.

Shogi is game like chess.

This software uses follwing technologies.

(1) Gated Recurrent Unit Policy Neural Network

(2) Convolutional Value Neural Network

(3) Monte Carlo Tree Search

(1) and (2) are written by PyTorch. (2) is written by C#.

## Neural Network

Gated Recurrent Unit Policy Neural Network predicts best move of current position. This Network does not use positional information, but uses timeline of moves. Convolutional Value Neural Network evaluates current position. This Network uses positional information.

### Input Features of Gated Recurrent Unit Policy Neural Network

...

### Convolutional Value Neural Network

(1) position of pieces

(2) effects of pieces

(3) difference of effects for every square

(4) turn

### Output Labels of Gated Recurrent Unit Neural Policy Neural Network

(1) position of move to

(2) move direction

The number of output labels are 32.

### Output Labels of Convolutional Value Neural Network

win or lost

## Learning functions

Learning functions are written by PyTorch. Policy Neural Network learns multi task classification. Value Neural Network learns binary classification.

### Record format for learning

Record format for learning is as below.

B,119,2726FU,3334FU,7776FU,4344FU, ...

First column is game result. Second Column is game ply. The following columns are moves. When learning Value Neural Network, you use first column. And learning Policy Neural Network, you use columns from third column to last column.

This software use the records created by Gikou-2.0.1.

## Operating environment

(1) OS: Windows 11 Pro

(2) Memory: 16GB or more. About 32GB is recommended.

(3) Memory usage on C# side: Less than 300MB when the MCTS task is 8.

(4) .NET Version: .NET 9.0

(5) Memory usage on the PyTorch side: About 3.6 GB. It is slightly heavy.

(6) Python's version: 3.12.10

(7) It's necessary that the latest stable version of PyTorch is installed.

(8) CUDA Version: 12.8.1

(9) The cuDNN corresponding to (8) must be installed. (*)

* As far as I know, the front-end version of cuDNN's license is changed to open source license, and that is installed using pip.

## Known bugs

(1) Output channels of Policy Neural Network is 256 channels. This must be 32 channels. This bug is only in Normal CNN Version, not in Lazy CNN Version.

(2) The table of correct answer label is not correct. But, after fixing this bug and re-learning, the accuracy was almost as same as before. So, I didn't fix this bug.

(3) On the play out function in MCTS, a illegal move is selected.

## Known problems

(1) On the play out function in MCTS, Rollout policy is not implemented.

(2) Compared to top-class software, the recognition accuracy of Policy Network is poor.

(3) This software does not support USI.

(4) If you create an executable file in a release build, you will get a deadlock when searching.

(5) When mate searching with multi-task, sometimes, this software doesn't recognize mate.

##How to build

Double click "Asklepios.sln" and build with using Visual Studio. I reccomend you debug build with x64 mode.

See "Known bugs" (4).

## Hou to use

If you navigate to the cnn folder or lazy_cnn folder and execute the start.bat, the specified game record will be analyzed. The command-line arguments are as follows:

(1) Date of the game

(2) Title name

(3) Black player name

(4) White player name

(5) Record file name for analyzing

(6) Output file name

(7) The number of tasks

(8) Thinking seconds per one move

## Contents of Release file

The execution environment of this software is contained in the Release file.

## References

I developed this software referring to the softwares as below.

(1) Bonanza

(2) Apery

(3) YaneuraOu

(4) Gikou

(5) dlshogi

As far as I know, the source code for Bonanza and dlshogi is currently not publicly available.

I developed this software referring to the books as below. All books are written in Japanese, so I write the name of the books in Japanese.

(1) 山岡忠夫(2018),『将棋AIで学ぶディープラーニング』マイナビ出版 

(2) 山岡忠夫、加納邦彦(2021), 『強い将棋ソフトの創りかた　Pythonで実装するディープラーニング将棋AI』マイナビ出版

(3) 大槻知史(著)、三宅陽一郎(監修)(2018), 『最強囲碁AI アルファ碁解体新書　増補改訂版』翔泳社

(4) 原田達也(2017), 機械学習プロフェッショナルシリーズ『画像認識』講談社
