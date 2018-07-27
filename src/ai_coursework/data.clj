(ns ai-coursework.data
  (:require [ai-coursework.train :as t]
            [clojure.data.csv :as csv]
            [clojure.java.io :as io]))

(defn numberize
  "Remove the headers of the csv and convert all
  the values to numbers."
  [csv]
  (map (fn [row]
         (map read-string row))
       (rest csv)))

(defn normalize
  "Normalize the csv values and produce a function that
  converts normalized outputs back to de-normalized values."
  [csv n]
  (let [min-max (reduce (fn [min-max row]
                          (map (fn [[mn mx] val]
                                 [(min mn val) (max mx val)])
                               min-max
                               row))
                        (map (fn [n] [n n]) (first csv))
                        (rest csv))]
    {:norm (reduce (fn [csv row]
                     (conj csv
                           (map (fn [[min max] val]
                                  (+ (/ (* 0.8
                                           (- val min))
                                        (- max min))
                                     0.1))
                                min-max
                                row)))
                   []
                   csv)
     :denorm-out (fn [output]
                   (map (fn [[min max] val]
                          (+ (/ (* (- val 0.1)
                                   (- max min))
                                0.8)
                             min))
                        (drop n min-max)
                        output))}))

(defn csv->examples
  "Transform csv data into a set of examples.
  Assumes all values are numbers.
  Assumes last value is only output."
  [csv n]
  (let [converted (numberize csv)
        {:keys [norm denorm-out]} (normalize converted n)]
    {:examples (map (fn [row]
                      {:input (take n row)
                       :expected (drop n row)})
                    norm)
     :denorm-out denorm-out}))

(defn random-split
  "Split the data into a number of ratios."
  [data ratios]
  (let [random (shuffle data)
        len (count data)
        idxs (map #(int (* len %)) ratios)]
    (loop [acc []
           idxs (butlast idxs)
           data random]
      (if (empty? idxs)
        (conj acc data)
        (recur (conj acc (take (first idxs) data))
               (rest idxs)
               (drop (first idxs) data))))))

(defn create-comparison
  "Given a network and a set of examples, create a list of pairs of modeled
  and actual outputs."
  [network examples denorm]
  (map (fn [{:keys [input expected]}]
         [(first (denorm (t/run-with-output network input)))
          (first (denorm expected))])
       examples))

(defn pairs->csv
  "Given a list of correct and modeled values, create a csv with headings
  `Modeled` and `Correct`."
  [paris file-path]
  (with-open [f (io/writer file-path)]
    (csv/write-csv f (cons ["Modeled" "Actual"] paris))))
