<?php
use Rubix\ML\Benchmarks\Tokenizers\WhitespaceBench;
use Rubix\ML\Loggers\Screen;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use Rubix\ML\Transformers\TfIdfTransformer;

require_once('vendor/autoload.php');

use Rubix\ML\Pipeline;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Tokenizers\WordStemmer;
use Sastrawi\Stemmer\StemmerFactory;
use Wamania\Snowball\StemmerFactory as WS;
use Rubix\ML\Extractors\Concatenator;
use Rubix\ML\Transformers\RegexFilter;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\StopWordFilter;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;


$logger = new Screen();

$logger->info('Loading CSV into memory');


/**
 * pastikan anda memiliki semua dataset dibawah yang valid formatnya
 * CVS dibawah berisi masing-masing 1000 record
 */
$extractor = new Concatenator([
    new CSV('dataset/scopus/art_humanities_1000.csv', true, ';'),
    new CSV('dataset/scopus/engineering_1000.csv', true, ';'),
    new CSV('dataset/scopus/life_science_medicine_1000.csv', true, ';'),
    new CSV('dataset/scopus/natural_science_1000.csv', true, ';'),
    new CSV('dataset/scopus/social_science_management_1000.csv', true, ';'),
]);


// Load dataset dari csv
$dataset = Labeled::fromIterator($extractor);

// cek dataset
// print_r($dataset);
// exit();

// membagi dataset menjadi data testing dan training
[$training, $testing] = $dataset->stratifiedSplit(0.8);

// print_r($testing);
// exit();

// fungsi untuk stemmer bahasa inggris dengan fungsi bawaan rubixml
$stemmeringgris = function (&$sample, $offset, $context) {
    $stemmer_en = new WordStemmer('english');

    $words = new Word();

    $arrwords = [];
    $arrsentence = [];
    foreach ($sample as $s) {
        if ($s !== null) {
            $array_words = $words->tokenize($s);

            foreach ($array_words as $aw) {
                if ($aw !== null) {
                    $stemmres = $stemmer_en->tokenize($aw);
                    $arrwords[] = $stemmres[0];
                }
            }

            // print_r($arrwords);
            $arrimploded = implode(" ", $arrwords);
            // var_dump($arrimploded);

        }
        $sample[0] = $arrimploded;
    }

    // $sample[0] = implode(" ", $stemmer_en->tokenize($sample));
    // var_dump( $sample[0]);

};


//fungsi stemmer bahasa indonesia dengan library sastrawi
// ON PROGRESS

//stopword-id untuk bhs indonesia, stopword-en untuk bhs inggris
$stopwords_id = file("stopword-en.txt", FILE_IGNORE_NEW_LINES);

$estimator = new Pipeline([
    new RegexFilter([
        RegexFilter::EXTRA_WHITESPACE,
        RegexFilter::EXTRA_WORDS
    ]),
    new StopWordFilter($stopwords_id),
    new TextNormalizer(),
    new LambdaFunction($stemmeringgris, 'stemmeringgris'),
    new WordCountVectorizer(),
], new KNearestNeighbors(3), true);

//proses training
$estimator->train($training);

// cek apakah proses training berhasil
print_r($estimator->trained());
// exit();

// testing menggunakan data testing dari dataset
$predictions = $estimator->predict($testing);

//hasil testing menggunakan data testing dari dataset
print_r($predictions);

// exit();

// validasi, confusion matrix, akurasi, dll
$report = new MulticlassBreakdown();

$results = $report->generate($predictions, $testing->labels());

echo $results;

/**
 * CATATAN:
 * - pastikan anda memiliki semua data .csv yang digunakan pada kode ini
 * - pilihlah stemmer untuk bahasa inggris dan bahasa indonesia, atau pilih dua2nya
 * - anda bisa membandingkan bandingkan hasil validasi dengan stemmer dan tanpa stemmer
 */