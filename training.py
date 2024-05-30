from PyQt5.QtCore import QThread, pyqtSignal
import torch
import torch.optim as optim
import torch.nn as nn
from resNet50 import ResNet, BasicBlock
from inceptionV1 import InceptionV1
from alexNet import AlexNet
from DenseNet import DenseNet

class TrainingThreadResnet(QThread):
    progress_updated = pyqtSignal(int)
    training_stopped = pyqtSignal()
    epoch_updated = pyqtSignal(int, float)
    batch_updated = pyqtSignal(int, int, float)
    train_loss_updated = pyqtSignal(list)
    val_accuracy_updated = pyqtSignal(list)

    def __init__(self, train_loader, test_loader, epochs):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self._is_running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=36).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

        train_losses = []
        val_accuracies = []
        self.final_train_losses = []  # Store final losses here
        self.final_val_accuracies = []  # Store final accuracies here

        for epoch in range(self.epochs):
            if not self._is_running:
                break
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                if not self._is_running:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.batch_updated.emit(epoch + 1, i + 1, running_loss / (i + 1))

            avg_train_loss = running_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.train_loss_updated.emit(train_losses)

            self.epoch_updated.emit(epoch + 1, avg_train_loss)
            self.progress_updated.emit(int((epoch + 1) / self.epochs * 100))

            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            self.val_accuracy_updated.emit(val_accuracies)
            self.final_train_losses.append(avg_train_loss)
            self.final_val_accuracies.append(accuracy)

        self.training_stopped.emit()

    def stop(self):
        self._is_running = False

class TrainingThreadInceptionV1(QThread):
    progress_updated = pyqtSignal(int)
    training_stopped = pyqtSignal()
    epoch_updated = pyqtSignal(int, float)
    batch_updated = pyqtSignal(int, int, float)
    train_loss_updated = pyqtSignal(list)
    val_accuracy_updated = pyqtSignal(list)

    def __init__(self, train_loader, test_loader, epochs):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self._is_running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        model = InceptionV1(num_classes=36).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        
        train_losses = []
        val_accuracies = []
        self.final_train_losses = []  # Store final losses here
        self.final_val_accuracies = []  # Store final accuracies here

        for epoch in range(self.epochs):
            if not self._is_running:
                break
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                if not self._is_running:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.batch_updated.emit(epoch + 1, i + 1, running_loss / (i + 1))

            avg_train_loss = running_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.train_loss_updated.emit(train_losses)

            self.epoch_updated.emit(epoch + 1, avg_train_loss)
            self.progress_updated.emit(int((epoch + 1) / self.epochs * 100))

            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            self.val_accuracy_updated.emit(val_accuracies)
            self.final_train_losses.append(avg_train_loss)
            self.final_val_accuracies.append(accuracy)

        self.training_stopped.emit()

    def stop(self):
        self._is_running = False


class TrainingThreadAlexNet(QThread):
    progress_updated = pyqtSignal(int)
    training_stopped = pyqtSignal()
    epoch_updated = pyqtSignal(int, float)
    batch_updated = pyqtSignal(int, int, float)
    train_loss_updated = pyqtSignal(list)
    val_accuracy_updated = pyqtSignal(list)

    def __init__(self, train_loader, test_loader, epochs):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self._is_running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        model = AlexNet(num_classes=36).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
        
        train_losses = []
        val_accuracies = []
        self.final_train_losses = []  # Store final losses here
        self.final_val_accuracies = []  # Store final accuracies here

        for epoch in range(self.epochs):
            if not self._is_running:
                break
            running_loss = 0.0
            for i, data in enumerate(self.train_loader, 0):
                if not self._is_running:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                self.batch_updated.emit(epoch + 1, i + 1, running_loss / (i + 1))

            avg_train_loss = running_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.train_loss_updated.emit(train_losses)

            self.epoch_updated.emit(epoch + 1, avg_train_loss)
            self.progress_updated.emit(int((epoch + 1) / self.epochs * 100))

            correct = 0
            total = 0
            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            self.val_accuracy_updated.emit(val_accuracies)
            self.final_train_losses.append(avg_train_loss)
            self.final_val_accuracies.append(accuracy)

        self.training_stopped.emit()

    def stop(self):
        self._is_running = False


class TrainingThreadDenseNet(QThread):
    progress_updated = pyqtSignal(int)
    training_stopped = pyqtSignal()
    epoch_updated = pyqtSignal(int, float)
    batch_updated = pyqtSignal(int, int, float)
    train_loss_updated = pyqtSignal(list)
    val_accuracy_updated = pyqtSignal(list)

    def __init__(self, train_loader, test_loader, epochs):
        super().__init__()
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.epochs = epochs
        self._is_running = True
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def run(self):
        model = DenseNet(num_classes=36).to(self.device)
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
        scaler = torch.cuda.amp.GradScaler()  # Mixed precision scaler

        train_losses = []
        val_accuracies = []
        self.final_train_losses = []  # Store final losses here
        self.final_val_accuracies = []  # Store final accuracies here

        for epoch in range(self.epochs):
            if not self._is_running:
                break
            running_loss = 0.0
            model.train()
            for i, data in enumerate(self.train_loader, 0):
                if not self._is_running:
                    break
                inputs, labels = data
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()
                with torch.cuda.amp.autocast():  # Mixed precision context
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()
                self.batch_updated.emit(epoch + 1, i + 1, running_loss / (i + 1))

            avg_train_loss = running_loss / len(self.train_loader)
            train_losses.append(avg_train_loss)
            self.train_loss_updated.emit(train_losses)

            self.epoch_updated.emit(epoch + 1, avg_train_loss)
            self.progress_updated.emit(int((epoch + 1) / self.epochs * 100))

            correct = 0
            total = 0
            model.eval()
            with torch.no_grad():
                for data in self.test_loader:
                    images, labels = data
                    images, labels = images.to(self.device), labels.to(self.device)
                    outputs = model(images)
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

            accuracy = 100 * correct / total
            val_accuracies.append(accuracy)
            self.val_accuracy_updated.emit(val_accuracies)
            self.final_train_losses.append(avg_train_loss)
            self.final_val_accuracies.append(accuracy)

        self.training_stopped.emit()

    def stop(self):
        self._is_running = False
