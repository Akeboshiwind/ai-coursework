(ns ai-coursework.core
  (:require [ai-coursework.layers :as layers]
            [ai-coursework.network :as network]
            [ai-coursework.train :as train]
            [ai-coursework.utils :as u]
            [ai-coursework.data :as d]
            [clojure.data.csv :refer :all]
            [clojure.tools.cli :refer [cli]]))

(defn main- [& args]
  (let [[opts args banner] (cli args
                                ["-h" "--help" "Print this help"
                                 :default false :flag true]
                                ["-i" "--input" "Input file"
                                 :default "Data.csv"]
                                ["-o" "--output" "Output file"
                                 :default "Output.csv"])]
    (if (:help opts)
      (println banner)
      (do
        ;;; Import the data
        ;; This step imports the data then normalizes it then creates examples out of it
        ;; It then outputs the examples as well as a function that given a normalized
        ;; output will produced a de-normalized output
        (println "Importing data...")
        (let [{:keys [examples denorm-out]}
              (-> (slurp (:input opts))
                  (read-csv)
                  (d/csv->examples 6))]
          (def data examples)
          (def denorm denorm-out))

        ;;; Split the data
        ;; This step randomizes the order of the data and then splits it into
        ;; three data sets
        ;; `train-set` is used to train the network
        ;; `test-set` is used to check if the network is overfitting the `train-set`
        ;; `validation-set` is used at the end to estimate the accuracy of the network
        (println "Splitting data...")
        (let [[train test val] (d/random-split data [0.6 0.2 0.2])]
          (def train-set train)
          (def test-set test)
          (def validation-set val))

        ;;; Describe the network
        ;; A network description is a list of layers
        ;; Each layer specifies how many neurons it has in it as well as what type of
        ;; neuron they are
        ;; It is assumed each of the layers is fully connected to the next layer
        (def network-desc
          [(layers/input 6)
           (layers/sigmoid 5)
           (layers/sigmoid 1)])

        ;;; Build the network
        ;; Build the network given the network description and the learning parameter
        (println "Building network...")
        (def network
          (network/build network-desc 0.1))

        ;;; Train the network
        ;; Train the network using the `train-set`
        ;; Stops either after 1000 epochs or if the estimated error of the network on
        ;; `test-set` goes up, whichever is first
        ;; Also prints the epoch number every 100 epochs
        (println "Training network...")
        (println "This will take a while")
        (def trained-network
          (train/train network train-set (train/combine
                                          (train/check-n-epochs 100)
                                          (train/max-epoch 1000)
                                          (train/early-stopping test-set 100)
                                          train/save-network)))

        ;;; Print the mean squared error of the network using the `validation-set`
        ;; This should give you a value to compare against other networks to gauge the
        ;; accuracy of the network
        (println "Network mean error: ")
        (clojure.pprint/pprint
         (train/network-mean-error
          trained-network
          validation-set))

        ;;; Output the comparison to a file "Output.csv"
        (println "Outputting to file...")
        (-> (d/create-comparison trained-network data denorm)
            (d/pairs->csv (:output opts)))

        (println "Finished!")))))
