#include "BackendDialog.hpp"
#include "BackendServiceManager.hpp"
#include "TrainingCurveWidget.hpp"

#include "application/application.h"

#include <QDir>
#include <QComboBox>
#include <QImage>
#include <QFile>
#include <QJsonArray>
#include <QJsonDocument>
#include <QJsonObject>
#include <QLabel>
#include <QLineEdit>
#include <QPlainTextEdit>
#include <QPushButton>
#include <QScrollBar>
#include <QTcpServer>
#include <QTcpSocket>
#include <QTemporaryDir>
#include <QSettings>
#include <QTest>
#include <QTreeWidget>

class FakeBackend final : public QObject {
    Q_OBJECT
public:
    QTcpServer server;
    int requestCount = 0;
    int submitCount = 0;
    int detailCount = 0;
    QJsonObject submitted;
    QByteArray authorization;

    FakeBackend()
    {
        connect(&server, &QTcpServer::newConnection, this, [this] {
            while (server.hasPendingConnections()) {
                QTcpSocket *socket = server.nextPendingConnection();
                connect(socket, &QTcpSocket::readyRead, socket, [this, socket] {
                    QByteArray buffer = socket->property("buffer").toByteArray();
                    buffer += socket->readAll();
                    socket->setProperty("buffer", buffer);
                    const qsizetype boundary = buffer.indexOf("\r\n\r\n");
                    if (boundary < 0) return;
                    const QByteArray headers = buffer.left(boundary);
                    const QByteArray firstLine = headers.left(headers.indexOf("\r\n"));
                    const QByteArray method = firstLine.left(firstLine.indexOf(' '));
                    const QByteArray path = firstLine.mid(firstLine.indexOf(' ') + 1,
                        firstLine.lastIndexOf(' ') - firstLine.indexOf(' ') - 1);
                    qsizetype contentLength = 0;
                    for (const QByteArray &line : headers.split('\n')) {
                        if (line.toLower().startsWith("authorization:")) authorization = line.mid(line.indexOf(':') + 1).trimmed();
                        if (line.toLower().startsWith("content-length:"))
                            contentLength = line.mid(line.indexOf(':') + 1).trimmed().toLongLong();
                    }
                    const QByteArray body = buffer.mid(boundary + 4);
                    if (body.size() < contentLength) return;
                    ++requestCount;
                    QByteArray responseBody;
                    int status = 200;
                    const QString existingJobId = QStringLiteral("550e8400-e29b-41d4-a716-446655440000");
                    if (path.endsWith("/health")) {
                        if (path.startsWith("/bad/")) {
                            status = 401;
                            responseBody = QByteArrayLiteral("{\"detail\":\"token rejected\"}");
                        } else if (path.startsWith("/unavailable/")) {
                            responseBody = QJsonDocument(QJsonObject{{QStringLiteral("status"), QStringLiteral("ok")},
                                {QStringLiteral("container"), QJsonObject{{QStringLiteral("available"), false},
                                    {QStringLiteral("runtime"), QStringLiteral("podman")},
                                    {QStringLiteral("error"), QStringLiteral("Container daemon is unavailable.")}}}})
                                .toJson(QJsonDocument::Compact);
                        } else {
                            responseBody = QJsonDocument(QJsonObject{{QStringLiteral("status"), QStringLiteral("ok")},
                                {QStringLiteral("container"), QJsonObject{{QStringLiteral("available"), true},
                                    {QStringLiteral("runtime"), QStringLiteral("podman")},
                                    {QStringLiteral("error"), QJsonValue(QJsonValue::Null)}}}})
                                .toJson(QJsonDocument::Compact);
                        }
                    } else if (method == "POST" && path.endsWith("/v1/jobs")) {
                        const QJsonDocument parsed = QJsonDocument::fromJson(body.left(contentLength));
                        submitted = parsed.object();
                        ++submitCount;
                        responseBody = job("submitted-1", "queued", QJsonObject{{QStringLiteral("epochs"), QJsonArray{}},
                            {QStringLiteral("test_loss"), QJsonValue(QJsonValue::Null)}});
                    } else if (path.endsWith("/v1/jobs")) {
                        QJsonArray jobs;
                        jobs.append(QJsonObject{{QStringLiteral("id"), existingJobId},
                            {QStringLiteral("status"), QStringLiteral("completed")},
                            {QStringLiteral("created_at"), QStringLiteral("2026-10-05T12:00:00Z")}});
                        jobs.append(QJsonObject{{QStringLiteral("id"), QStringLiteral("legacy-1")},
                            {QStringLiteral("status"), QStringLiteral("completed")},
                            {QStringLiteral("created_at"), QStringLiteral("2026-10-04T12:00:00Z")}});
                        responseBody = QJsonDocument(QJsonObject{{QStringLiteral("jobs"), jobs}}).toJson(QJsonDocument::Compact);
                    } else if (path.contains("/v1/jobs/")) {
                        ++detailCount;
                        if (path.endsWith("/legacy-1")) {
                            QJsonArray legacyEpochs{QJsonObject{{QStringLiteral("epoch"), 3},
                                {QStringLiteral("training_loss"), 0.3}, {QStringLiteral("validation_loss"), 0.2}}};
                            responseBody = job("legacy-1", "completed", QJsonObject{
                                {QStringLiteral("steps"), QJsonArray{}}, {QStringLiteral("epochs"), legacyEpochs},
                                {QStringLiteral("test_loss"), 0.25}});
                        } else {
                        QJsonArray epochs;
                        for (int epoch = 1; epoch <= 120; ++epoch)
                            epochs.append(QJsonObject{{QStringLiteral("epoch"), epoch},
                                {QStringLiteral("training_loss"), 0.5}, {QStringLiteral("validation_loss"), 0.4}});
                        QJsonArray steps;
                        for (int step = 1; step <= 120; ++step)
                            steps.append(QJsonObject{{QStringLiteral("step"), step}, {QStringLiteral("epoch"), 1},
                                {QStringLiteral("training_loss"), step == 1 ? 0.6 : 0.5},
                                {QStringLiteral("validation_loss"), step == 1 ? 0.5 : 0.4}});
                        responseBody = job(existingJobId, "completed", QJsonObject{{QStringLiteral("epochs"), epochs},
                            {QStringLiteral("steps"), steps},
                            {QStringLiteral("test_loss"), 0.25}});
                        }
                    } else responseBody = QByteArrayLiteral("{}");
                    const QByteArray statusText = status == 200 ? QByteArrayLiteral("200 OK") : QByteArrayLiteral("401 Unauthorized");
                    socket->write("HTTP/1.1 " + statusText + "\r\nContent-Type: application/json\r\nContent-Length: " +
                        QByteArray::number(responseBody.size()) + "\r\nConnection: close\r\n\r\n" + responseBody);
                    socket->disconnectFromHost();
                });
            }
        });
    }

    static QByteArray job(const QString &id, const QString &status, const QJsonObject &metrics)
    {
        return QJsonDocument(QJsonObject{{QStringLiteral("id"), id},
            {QStringLiteral("status"), status},
            {QStringLiteral("created_at"), QStringLiteral("2026-10-05T12:00:00Z")},
            {QStringLiteral("error"), QJsonValue(QJsonValue::Null)},
            {QStringLiteral("metrics"), metrics}}).toJson(QJsonDocument::Compact);
    }
};

class BackendDialogTest final : public QObject {
    Q_OBJECT
private slots:
    void localBackendConfigurationValidationAndPersistence()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        QSettings::setDefaultFormat(QSettings::IniFormat);
        QSettings::setPath(QSettings::IniFormat, QSettings::UserScope, temporary.path());
        QTemporaryDir source;
        BackendServiceManager manager(source.path());
        BackendDialog dialog(nullptr, {}, {}, nullptr, &manager);
        const QStringList launcherControls{
            QStringLiteral("backendExecutor"), QStringLiteral("backendJobRoot"),
            QStringLiteral("backendDockerRuntime"), QStringLiteral("backendDockerImage"),
            QStringLiteral("backendSlurmHost"), QStringLiteral("backendSlurmRoot"),
            QStringLiteral("backendSlurmImage"), QStringLiteral("backendSlurmPartition"),
            QStringLiteral("backendSlurmCpus"), QStringLiteral("backendSlurmMemory"),
            QStringLiteral("backendSlurmTime"), QStringLiteral("backendManagedStart")};
        for (const QString &name : launcherControls)
            QVERIFY2(!dialog.findChild<QWidget *>(name), qPrintable(name));
        dialog.show();
        dialog.findChild<QLineEdit *>(QStringLiteral("backendToken"))->setText(QStringLiteral("session-secret"));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConfigure")), Qt::LeftButton);
        auto *configuration = dialog.findChild<QDialog *>(QStringLiteral("backendConfigurationWindow"));
        QVERIFY(configuration);
        QVERIFY(configuration != &dialog);
        QVERIFY(configuration->isWindow());
        QVERIFY(!configuration->isModal());
        QVERIFY(dialog.isEnabled());
        QVERIFY(configuration->isVisible());
        auto *executor = configuration->findChild<QComboBox *>(QStringLiteral("backendExecutor"));
        auto *jobRoot = configuration->findChild<QLineEdit *>(QStringLiteral("backendJobRoot"));
        auto *startButton = configuration->findChild<QPushButton *>(QStringLiteral("backendManagedStart"));
        QVERIFY(executor);
        QVERIFY(jobRoot);
        QVERIFY(startButton);
        QCOMPARE(executor->parentWidget(), configuration);
        QCOMPARE(jobRoot->parentWidget(), configuration);
        executor->setCurrentIndex(1);
        jobRoot->setText(QStringLiteral("relative/jobs"));
        QTest::mouseClick(startButton, Qt::LeftButton);
        QVERIFY(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationStatus"))->text().contains(
            QStringLiteral("absolute path")));

        jobRoot->setText(temporary.path());
        configuration->findChild<QLineEdit *>(QStringLiteral("backendSlurmRoot"))->setText(QStringLiteral("/cluster/jobs"));
        configuration->findChild<QLineEdit *>(QStringLiteral("backendSlurmImage"))->setText(QStringLiteral("/cluster/images/worker.sif"));
        QTest::mouseClick(startButton, Qt::LeftButton);
        QVERIFY(configuration->findChild<QLabel *>(QStringLiteral("backendConfigurationStatus"))->text().contains(
            QStringLiteral("source tree and uv workspace")));
        QVERIFY(!manager.isRunning());

        QSettings saved;
        saved.beginGroup(QStringLiteral("backendLauncher"));
        QCOMPARE(saved.value(QStringLiteral("executor")).toString(), QStringLiteral("slurm"));
        QVERIFY(!saved.contains(QStringLiteral("token")));
        QVERIFY(!saved.contains(QStringLiteral("bearerToken")));
    }

    void curvePathsDoNotFillBetweenSeries()
    {
        TrainingCurveWidget curve;
        curve.resize(640, 400);
        curve.setMetrics(QJsonObject{{QStringLiteral("steps"), QJsonArray{
            QJsonObject{{QStringLiteral("step"), 1}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 1.0}},
            QJsonObject{{QStringLiteral("step"), 2}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 3.0}},
            QJsonObject{{QStringLiteral("step"), 3}, {QStringLiteral("training_loss"), 0.5},
                        {QStringLiteral("validation_loss"), 1.0}}
        }}});
        curve.show();
        QCoreApplication::processEvents();
        QImage image(curve.size(), QImage::Format_ARGB32_Premultiplied);
        auto renderAndCheck = [&] {
            image.fill(Qt::transparent);
            curve.render(&image);
            QCOMPARE(image.pixelColor(200, 200), QColor(Qt::white));
            bool foundTrainingSeries = false;
            for (int y = 0; y < image.height() && !foundTrainingSeries; ++y) {
                for (int x = 0; x < image.width(); ++x) {
                    if (image.pixelColor(x, y) == QColor(QStringLiteral("#2677b8"))) {
                        foundTrainingSeries = true;
                        break;
                    }
                }
            }
            QVERIFY(foundTrainingSeries);
        };

        renderAndCheck();
        curve.setValidationVisible(false);
        renderAndCheck();
        curve.setValidationVisible(true);
        renderAndCheck();
    }

    void curveSupportsStepsFallbackInvalidAndLogScale()
    {
        TrainingCurveWidget curve;
        curve.resize(720, 360);
        QImage image(curve.size(), QImage::Format_ARGB32_Premultiplied);
        image.fill(Qt::transparent);
        curve.render(&image);
        QVERIFY(curve.trainingPoints().isEmpty());
        QVERIFY(!curve.usesStepAxis());
        QVERIFY(curve.xAxisLabel().contains(QStringLiteral("Epoch")));

        curve.setMetrics(QJsonObject{{QStringLiteral("steps"), QJsonArray{
            QJsonObject{{QStringLiteral("step"), 2}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 2.0}, {QStringLiteral("validation_loss"), 1.0}},
            QJsonObject{{QStringLiteral("step"), 4}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 0}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 2.5}, {QStringLiteral("epoch"), 1},
                        {QStringLiteral("training_loss"), 1.0}, {QStringLiteral("validation_loss"), 0.5}},
            QJsonObject{{QStringLiteral("step"), 6}, {QStringLiteral("epoch"), 2},
                        {QStringLiteral("training_loss"), 0.5}, {QStringLiteral("validation_loss"), -0.1}},
            QJsonObject{{QStringLiteral("step"), 7}, {QStringLiteral("epoch"), 2},
                        {QStringLiteral("training_loss"), QStringLiteral("NaN")}, {QStringLiteral("validation_loss"), 0.2}}
        }}});
        QVERIFY(curve.usesStepAxis());
        QCOMPARE(curve.xAxisLabel(), QStringLiteral("Optimizer step"));
        QCOMPARE(curve.trainingPoints().size(), 3);
        QCOMPARE(curve.validationPoints().size(), 3);
        QVERIFY(curve.dataNotice().contains(QStringLiteral("3 invalid")));
        curve.setLogarithmic(true);
        QVERIFY(curve.isLogarithmic());
        QVERIFY(curve.dataNotice().contains(QStringLiteral("non-positive")));
        curve.render(&image);

        curve.setMetrics(QJsonObject{{QStringLiteral("epochs"), QJsonArray{
            QJsonObject{{QStringLiteral("epoch"), 3}, {QStringLiteral("training_loss"), 0.7},
                        {QStringLiteral("validation_loss"), 0.6}}
        }}});
        QVERIFY(!curve.usesStepAxis());
        QCOMPARE(curve.trainingPoints().size(), 1);
        curve.render(&image);
    }

    void submitSnapshotJobsMetricsAndErrors()
    {
        QTemporaryDir temporary;
        QVERIFY(temporary.isValid());
        FakeBackend backend;
        QVERIFY(backend.server.listen(QHostAddress::LocalHost));
        NNApplication *application = nn_app_new(NN_SOURCE_DIR "/stereotype-packages/core");
        QVERIFY(application);
        char error[512] = {};
        QVERIFY2(nn_app_create(application, temporary.path().toUtf8().constData(), "ui-test", "UI Test", false,
                               error, sizeof(error)), error);
        const QByteArray dataset = QByteArrayLiteral("{\"name\":\"Test data\",\"batch\":{\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[1]}},\"targets\":{}}}");
        QVERIFY2(nn_app_create_dataset(application, "test.data", "0.1.0", dataset.constData(), false,
                                       error, sizeof(error)), error);
        const QString projectDirectory = QString::fromUtf8(nn_project_directory(nn_app_project(application)));
        const QString datasetManifestPath = QDir(projectDirectory).filePath(QStringLiteral("datasets/test.data-0.1.0/manifest.json"));
        QFile datasetManifestFile(datasetManifestPath);
        QVERIFY(datasetManifestFile.open(QIODevice::ReadOnly));
        QJsonObject datasetManifest = QJsonDocument::fromJson(datasetManifestFile.readAll()).object();
        datasetManifestFile.close();
        datasetManifest.insert(QStringLiteral("inferenceAssets"), QJsonArray{QStringLiteral("assets/vocab.txt")});
        QVERIFY(datasetManifestFile.open(QIODevice::WriteOnly | QIODevice::Truncate));
        QCOMPARE(datasetManifestFile.write(QJsonDocument(datasetManifest).toJson(QJsonDocument::Indented)),
                 qint64(QJsonDocument(datasetManifest).toJson(QJsonDocument::Indented).size()));
        datasetManifestFile.close();
        QVERIFY(QDir().mkpath(QDir(projectDirectory).filePath(QStringLiteral("assets"))));
        QFile vocabulary(QDir(projectDirectory).filePath(QStringLiteral("assets/vocab.txt")));
        QVERIFY(vocabulary.open(QIODevice::WriteOnly));
        QCOMPARE(vocabulary.write("digits\n"), qint64(7));
        vocabulary.close();
        QVERIFY(QDir().mkpath(QDir(projectDirectory).filePath(QStringLiteral("data"))));
        QFile sample(QDir(projectDirectory).filePath(QStringLiteral("data/sample.csv")));
        QVERIFY(sample.open(QIODevice::WriteOnly));
        QCOMPARE(sample.write("x,y\n1,2\n"), qint64(8));
        sample.close();

        BackendDialog dialog(application, [application] {
            char saveError[512] = {};
            return nn_app_save(application, saveError, sizeof(saveError));
        }, [](const QString &) { return false; });
        auto *endpoint = dialog.findChild<QLineEdit *>(QStringLiteral("backendEndpoint"));
        endpoint->setText(QStringLiteral("http://127.0.0.1:%1").arg(backend.server.serverPort()));
        dialog.findChild<QLineEdit *>(QStringLiteral("backendToken"))->setText(QStringLiteral("secret-for-test"));
        dialog.show();
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().startsWith(
            QStringLiteral("Connected — podman is ready for training.")) || dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().startsWith(
            QStringLiteral("Backend request completed.")), 3000);
        auto *jobs = dialog.findChild<QTreeWidget *>(QStringLiteral("backendJobs"));
        QTRY_COMPARE_WITH_TIMEOUT(jobs->topLevelItemCount(), 2, 3000);
        const QString existingJobId = jobs->topLevelItem(0)->data(0, Qt::UserRole).toString();
        QVERIFY(jobs->topLevelItem(0)->text(0).contains(QStringLiteral("2026-10-05 12:00")));
        QVERIFY(jobs->topLevelItem(0)->text(0).contains(existingJobId.right(8)));
        QVERIFY(!jobs->topLevelItem(0)->text(0).contains(existingJobId));
        QVERIFY(jobs->topLevelItem(0)->toolTip(0).contains(existingJobId));
        QVERIFY(jobs->topLevelItem(0)->toolTip(0).contains(QStringLiteral("2026-10-05T12:00:00Z")));
        QCOMPARE(jobs->columnCount(), 2);
        jobs->setCurrentItem(jobs->topLevelItem(0));
        auto *metrics = dialog.findChild<QPlainTextEdit *>(QStringLiteral("backendMetrics"));
        QTRY_VERIFY_WITH_TIMEOUT(metrics->toPlainText().contains(QStringLiteral("Final test loss: 0.25")), 3000);
        QVERIFY(metrics->toPlainText().contains(QStringLiteral("Step 1 (epoch 1) — training 0.6, validation 0.5")));
        auto *curve = dialog.findChild<TrainingCurveWidget *>(QStringLiteral("trainingCurve"));
        QVERIFY(curve->usesStepAxis());
        QCOMPARE(curve->trainingPoints().size(), 120);
        QCOMPARE(dialog.findChild<QLabel *>(QStringLiteral("finalTestLoss"))->text(), QStringLiteral("Final test loss: 0.25"));
        QVERIFY(metrics->verticalScrollBar()->maximum() > 0);
        metrics->verticalScrollBar()->setValue(metrics->verticalScrollBar()->maximum() / 2);
        const int scrollPosition = metrics->verticalScrollBar()->value();
        const int previousDetailCount = backend.detailCount;
        QTRY_VERIFY_WITH_TIMEOUT(backend.detailCount > previousDetailCount, 5000);
        QTest::qWait(100);
        QCOMPARE(metrics->verticalScrollBar()->value(), scrollPosition);

        jobs->setCurrentItem(jobs->topLevelItem(1));
        QTRY_VERIFY_WITH_TIMEOUT(metrics->toPlainText().contains(QStringLiteral("Epoch 3 — training 0.3, validation 0.2")), 3000);
        QVERIFY(!curve->usesStepAxis());
        QCOMPARE(curve->xAxisLabel(), QStringLiteral("Epoch"));
        QCOMPARE(curve->trainingPoints().size(), 1);

        dialog.findChild<QLineEdit *>(QStringLiteral("trainingEpochs"))->setText(QStringLiteral("7"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingBatchSize"))->setText(QStringLiteral("8"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingLearningRate"))->setText(QStringLiteral("0.02"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingSeed"))->setText(QStringLiteral("42"));
        dialog.findChild<QLineEdit *>(QStringLiteral("trainingPublishEverySteps"))->setText(QStringLiteral("25"));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendSubmit")), Qt::LeftButton);
        QTRY_COMPARE_WITH_TIMEOUT(backend.submitCount, 1, 5000);
        const QJsonObject training = backend.submitted.value(QStringLiteral("training")).toObject();
        QCOMPARE(training.value(QStringLiteral("epochs")).toInt(), 7);
        QCOMPARE(training.value(QStringLiteral("batch_size")).toInt(), 8);
        QCOMPARE(training.value(QStringLiteral("learning_rate")).toDouble(), 0.02);
        QCOMPARE(training.value(QStringLiteral("seed")).toInt(), 42);
        QCOMPARE(training.value(QStringLiteral("publish_every_steps")).toInt(), 25);
        QCOMPARE(backend.authorization, QByteArrayLiteral("Bearer secret-for-test"));
        const QJsonObject project = backend.submitted.value(QStringLiteral("project")).toObject();
        QVERIFY(project.value(QStringLiteral("manifest")).toObject().value(QStringLiteral("schemaVersion")).isDouble());
        QVERIFY(project.value(QStringLiteral("nodes")).isArray());
        const QJsonObject files = backend.submitted.value(QStringLiteral("files")).toObject();
        const QByteArray datasetCode = QByteArray::fromBase64(files.value(QStringLiteral("datasets/test.data-0.1.0/dataset.py")).toString().toLatin1());
        QVERIFY(datasetCode.contains("NotImplementedError"));
        QVERIFY(!QString::fromUtf8(QJsonDocument(backend.submitted).toJson()).contains(QStringLiteral("secret-for-test")));
        QCOMPARE(QByteArray::fromBase64(files.value(QStringLiteral("data/sample.csv")).toString().toLatin1()), QByteArray("x,y\n1,2\n"));
        QCOMPARE(QByteArray::fromBase64(files.value(QStringLiteral("assets/vocab.txt")).toString().toLatin1()), QByteArray("digits\n"));
        QVERIFY(!files.contains(QStringLiteral("data/.DS_Store")));
        QVERIFY(!files.contains(QStringLiteral("model.json")));

        endpoint->setText(QStringLiteral("http://127.0.0.1:%1/bad").arg(backend.server.serverPort()));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("token rejected")), 3000);
        endpoint->setText(QStringLiteral("http://127.0.0.1:%1/unavailable").arg(backend.server.serverPort()));
        QTest::mouseClick(dialog.findChild<QPushButton *>(QStringLiteral("backendConnect")), Qt::LeftButton);
        QTRY_VERIFY_WITH_TIMEOUT(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("training is unavailable")), 3000);
        QVERIFY(dialog.findChild<QLabel *>(QStringLiteral("backendStatus"))->text().contains(
            QStringLiteral("Container daemon is unavailable.")));
    }
};

QTEST_MAIN(BackendDialogTest)
#include "qt_backend_test.moc"
