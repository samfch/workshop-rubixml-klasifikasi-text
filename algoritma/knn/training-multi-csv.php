<?php
require_once('vendor/autoload.php');
use Rubix\ML\CrossValidation\Reports\AggregateReport;
use Rubix\ML\CrossValidation\Reports\ConfusionMatrix;
use Rubix\ML\Transformers\MultibyteTextNormalizer;
use Rubix\ML\Tokenizers\Word;
use Rubix\ML\Pipeline;
use Rubix\ML\Extractors\CSV;
use Rubix\ML\Datasets\Labeled;
use Rubix\ML\Tokenizers\WordStemmer;
use Rubix\ML\Transformers\RegexFilter;
use Rubix\ML\Transformers\LambdaFunction;
use Rubix\ML\Transformers\StopWordFilter;
use Rubix\ML\Transformers\TextNormalizer;
use Rubix\ML\Classifiers\KNearestNeighbors;
use Rubix\ML\Transformers\WordCountVectorizer;
use Rubix\ML\CrossValidation\Reports\MulticlassBreakdown;
use Rubix\ML\PersistentModel;
use Rubix\ML\Persisters\Filesystem;
use Rubix\ML\Extractors\Concatenator;


/**
 * pastikan anda memiliki semua dataset dibawah yang valid formatnya
 * CVS dibawah berisi masing-masing 1000 record
 */
$extractor = new Concatenator([
    new CSV('./dataset/scopus/art_humanities_1000.csv', true, ';'),
    // new CSV('./dataset/scopus/engineering_1000.csv', true, ';'),
    // new CSV('./dataset/scopus/life_science_medicine_1000.csv', true, ';'),
    // new CSV('./dataset/scopus/natural_science_1000.csv', true, ';'),
    // new CSV('./dataset/scopus/social_science_management_1000.csv', true, ';'),
]);

// Load dataset dari csv
$dataset = Labeled::fromIterator($extractor);

// membagi dataset menjadi data testing dan training
[$training, $testing] = $dataset->stratifiedSplit(0.7);

// fungsi untuk stemmer bahasa inggris
$stemmeringgris = function (&$sample, $offset, $context) {
    $stemmer_en = new WordStemmer('english');
    $words = new Word();
    $arrwords = [];
    foreach ($sample as $s) {
        if ($s !== null) {
            $array_words = $words->tokenize($s);

            foreach ($array_words as $aw) {
                if ($aw !== null) {
                    $stemmres = $stemmer_en->tokenize($aw);
                    $arrwords[] = $stemmres[0];
                }
            }
            $arrimploded = implode(" ", $arrwords);
        }
        $sample[0] = $arrimploded;
    }
};

// fungsi untuk stemmer bahasa indonesia
$stemmerindo = function (&$sample, $offset, $context) {
    $stemmerFactory = new \Sastrawi\Stemmer\StemmerFactory();
    $stemmer_id = $stemmerFactory->createStemmer();

    foreach ($sample as $s) {
        $stemmed = $stemmer_id->stem($s);
        $sample[0] = $stemmed;
    }
};


// stemming datasets (pilih sesuai bahasa)
$dataset->apply(new LambdaFunction($stemmeringgris, 'stemmer'));
// $dataset->apply(new LambdaFunction($stemmerindo,'stemmer'));

//daftar stopword sesuai bahasa
$stopwords_en = file("././stopword-en.txt", FILE_IGNORE_NEW_LINES);
// $stopwords_id = file("stopword-id.txt", FILE_IGNORE_NEW_LINES);

// pipeline untuk preprocessing dan klasifikasi
$estimator = new PersistentModel(
    new Pipeline([
        //PREPROCESSING
        new RegexFilter([
            RegexFilter::EXTRA_WHITESPACE,
            RegexFilter::EXTRA_WORDS
        ]), //text cleaning
        new MultibyteTextNormalizer(false), //case folding - lower case
        new StopWordFilter($stopwords_en), //stopwords
        new WordCountVectorizer(), //tokenizer (hapus untuk Bayes)

        // ALGORITMA KLASIFIKASI
    ], new KNearestNeighbors(3)),
    // nama file model
    new Filesystem('./knn/model-disimpan-knn.model')
    );

//proses training dataset
$estimator->train($training);

// cek apakah proses training berhasil
if ($estimator->trained()) {
    print_r('Training model....  ');

    // simpan model
    $estimator->save();
    print_r('Model disimpan.... OK  ');

}

// testing menggunakan data testing dari dataset
$predictions = $estimator->predict($testing);

// validasi, confusion matrix, akurasi, dll
$report = new AggregateReport([
    'breakdown' => new MulticlassBreakdown(),
    'confussion_matrix' => new ConfusionMatrix(),
]);

$results = $report->generate($predictions, $testing->labels());

echo $results;