package com.zatsepin.effective.ml.samples.bayes

typealias FeatureMapper = (String) -> List<String>

class BayesConfiguration(
        val featureMapper: FeatureMapper
)

data class FeatureKey(
        val targetClass: String,
        val feature: String
)

data class Classifier(
        val classFreq: Map<String, Double>,
        val featureFreq: Map<FeatureKey, Double>
)

data class Sample(
        val word: String,
        val targetClass: String
)


class NaiveBayesClassifier(
        private val config: BayesConfiguration
) {

    fun train(samples: List<Sample>): Classifier {
        val featureFreq: MutableMap<FeatureKey, Double> = mutableMapOf()
        val classFreq: MutableMap<String, Double> = mutableMapOf()
        samples.forEach { sample ->
            classFreq.increment(sample.targetClass)
            val features = getFeatures(sample.word)
            features.forEach { feature ->
                featureFreq.increment(FeatureKey(sample.targetClass, feature))
            }
        }
        featureFreq.normalize { key -> classFreq[key.targetClass]!! }
        classFreq.normalize(samples.size.toDouble())
        return Classifier(classFreq, featureFreq)
    }

    fun classify(word: String, classifier: Classifier): String {
        val features = getFeatures(word)
        val classProbabilities = classifier.classFreq
                .map { targetClass ->
                    //sum(-ln(P(O|C))
                    val featuresProb = features
                            .asSequence()
                            .map { feature ->
                                val key = FeatureKey(targetClass.key, feature)
                                val rawValue = classifier.featureFreq[key] ?: Math.pow(10.0, -18.0)
                                -Math.log(rawValue)
                            }
                            .sum()
                    //-ln(P(C))
                    val classProb = -Math.log(targetClass.value)
                    val totalProb = classProb + featuresProb
                    targetClass.key to totalProb
                }
                .sortedBy { it.second }
        return classProbabilities.first().first
    }

    private fun <T> MutableMap<T, Double>.normalize(n: Double) {
        this.normalize { _ -> n }
    }

    private fun <T> MutableMap<T, Double>.normalize(n: (key: T) -> Double) {
        this.replaceAll { key, value -> value / n(key) }
    }

    private fun <T : Any> MutableMap<T, Double>.increment(key: T) {
        this[key] = (this[key] ?: 0.0) + 1
    }

    private fun getFeatures(word: String): List<String> {
        return config.featureMapper(word)
    }

}