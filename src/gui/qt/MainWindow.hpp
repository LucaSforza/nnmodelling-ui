#ifndef NN_GUI_MAIN_WINDOW_HPP
#define NN_GUI_MAIN_WINDOW_HPP

#include <QMainWindow>
#include <memory>

class QAction;
class QCheckBox;
class QCloseEvent;
class QLineEdit;
class QTreeWidget;
class QTreeWidgetItem;
class QTimer;
class QToolButton;
class QWidget;
class NNApplication;
struct NNAutomation;
class GraphScene;
class GraphView;
class BackendServiceManager;
enum class FlowDirection;

class MainWindow final : public QMainWindow {
    Q_OBJECT
public:
    explicit MainWindow(NNApplication *application, QWidget *parent = nullptr);
    ~MainWindow() override;

    bool openProject(const QString &directory);
    void showProjectChooser();
    bool startAutomation(const QString &socketPath);

protected:
    void closeEvent(QCloseEvent *event) override;

private:
    void buildUi();
    void refreshAll();
    void refreshPalette();
    void refreshInspector();
    void refreshResources();
    void refreshDiagnostics();
    bool revealNode(const QString &id);
    void updateWindowTitle();
    bool confirmReplaceProject();
    bool saveProject();
    void createProject(bool mnist);
    void createLlmProject();
    void addSelectedPackage();
    void arrangeCurrentScope(FlowDirection direction);
    void restoreEdit(bool redo);
    void editNodeName(const QString &nodeId, const QString &name);
    void editNodeParameter(const QString &nodeId, const QString &key, const QString &value);
    void selectResource(QTreeWidgetItem *item, int column);
    void createStereotype();
    void createDataset(const QString &id = QString(), const QString &version = QString());
    void manageDatasets();
    void manageOperations();
    void selectDataset(QTreeWidgetItem *item, int column);
    static char *automationUiCallback(void *user, const char *operation,
                                     const char *argsJson, char *error, size_t cap);

    std::unique_ptr<NNApplication, void (*)(NNApplication *)> application_;
    GraphScene *scene_ = nullptr;
    GraphView *view_ = nullptr;
    QWidget *paletteTree_ = nullptr;
    QLineEdit *paletteSearch_ = nullptr;
    QTreeWidget *inspector_ = nullptr;
    QTreeWidget *resources_ = nullptr;
    QTreeWidget *diagnostics_ = nullptr;
    QCheckBox *currentScopeProblems_ = nullptr;
    QToolButton *scopeSelector_ = nullptr;
    QTreeWidget *scopeTree_ = nullptr;
    QAction *undoAction_ = nullptr;
    QAction *redoAction_ = nullptr;
    QTimer *automationTimer_ = nullptr;
    NNAutomation *automation_ = nullptr;
    BackendServiceManager *backendService_ = nullptr;
    QString selectedPaletteId_;
    bool refreshing_ = false;
};

#endif
