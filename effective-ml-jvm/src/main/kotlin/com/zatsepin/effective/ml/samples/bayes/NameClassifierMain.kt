package com.zatsepin.effective.ml.samples.bayes

import com.fasterxml.jackson.annotation.JsonIgnoreProperties
import com.fasterxml.jackson.core.type.TypeReference
import com.fasterxml.jackson.databind.ObjectMapper
import com.fasterxml.jackson.module.kotlin.registerKotlinModule
import java.math.RoundingMode


@JsonIgnoreProperties(ignoreUnknown = true)
data class Name(
        val name: String,
        val gender: String
)

fun Name.toSample(): Sample {
    return Sample(name, gender)
}

const val EN_NAMES_PATH = "/samples/bayes/names/samples_england.json"
const val CN_NAMES_PATH = "/samples/bayes/names/samples_canada.json"
const val GE_NAMES_PATH = "/samples/bayes/names/samples_germany.json"
const val US_NAMES_PATH = "/samples/bayes/names/samples_usa.json"
const val AU_NAMES_PATH = "/samples/bayes/names/samples_australia.json"
const val IN_NAMES_PATH = "/samples/bayes/names/samples_india.json"

val usedPaths = listOf(
        EN_NAMES_PATH,
        CN_NAMES_PATH,
        GE_NAMES_PATH,
        US_NAMES_PATH,
        AU_NAMES_PATH,
        IN_NAMES_PATH
)
const val TESTS_PATH = "/samples/bayes/names/tests.json"
fun main(args: Array<String>) {
    val bayes = NaiveBayesClassifier(
            BayesConfiguration { sample ->
                listOf(
                         sample[0].toString(),//first letter
                         sample[1].toString(),//second letter
                         //sample[sample.length - 1].toString(),//last letter
                         //sample[0].toString()
                        //       + sample[1].toString(),//first and second letters
                        // sample[sample.length - 1].toString()
                        //   + sample[sample.length - 2].toString(),//two last letters,
                        sample[0].toString()
                                + sample[1].toString()
                                + sample[sample.length - 2].toString()
                                + sample[sample.length - 1].toString()
                )
            }
    )
    val samples = getSamplesFromNames(*usedPaths.toTypedArray())
    val classifier = bayes.train(samples)

    val oneResult = bayes.classify("Sophia", classifier)
    println(oneResult)

    val tests = getSamplesFromNames(*usedPaths.toTypedArray())
    val result = classifyAll(bayes, tests, classifier)
    println("Error ${(result * 100).toBigDecimal().setScale(2, RoundingMode.UP)}%")
}

fun getSamplesFromNames(vararg resourcePaths: String): List<Sample> {
    val objectMapper = ObjectMapper().registerKotlinModule()
    return resourcePaths.flatMap {
        objectMapper.readValue<List<Name>>(
                Name::class.java.getResource(it),
                object : TypeReference<List<Name>>() {}
        ).map(Name::toSample)
    }
}


/**
 * @return error percentages
 */
fun classifyAll(bayes: NaiveBayesClassifier, tests: List<Sample>, classifier: Classifier): Double {
    var error = 0
    tests.forEach { test ->
        val result = bayes.classify(test.word, classifier)
        if (result != test.targetClass) {
            error += 1
        }
    }
    return error / tests.size.toDouble()
}
