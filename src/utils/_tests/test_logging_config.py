#!/usr/bin/env python3
"""
Unit tests for src/utils/logging_config.py - Logging Configuration
"""

import pytest
import logging
import os
from unittest.mock import patch, mock_open, Mock
import tempfile
import shutil

from src.utils.logging_config import setup_logging, logger


class TestSetupLogging:
    """Test setup_logging function."""
    
    @patch('src.utils.logging_config.os.makedirs')
    @patch('src.utils.logging_config.logging.basicConfig')
    def test_setup_logging_basic(self, mock_basic_config, mock_makedirs):
        """Test basic logging setup."""
        result_logger = setup_logging()
        
        # Should create logs directory
        mock_makedirs.assert_called_once()
        
        # Should configure logging
        mock_basic_config.assert_called_once()
        
        # Should return a logger
        assert isinstance(result_logger, logging.Logger)
    
    @patch('src.utils.logging_config.LOGS_DIR', 'test_logs')
    @patch('src.utils.logging_config.os.makedirs')
    @patch('src.utils.logging_config.logging.basicConfig')
    def test_setup_logging_with_custom_logs_dir(self, mock_basic_config, mock_makedirs):
        """Test logging setup with custom logs directory."""
        setup_logging()
        
        # Should create the custom logs directory
        mock_makedirs.assert_called_with('test_logs', exist_ok=True)
    
    @patch('src.utils.logging_config.os.makedirs')
    @patch('src.utils.logging_config.logging.basicConfig')
    def test_setup_logging_configuration_parameters(self, mock_basic_config, mock_makedirs):
        """Test that logging is configured with correct parameters."""
        setup_logging()
        
        # Check basicConfig call arguments
        call_args = mock_basic_config.call_args
        kwargs = call_args.kwargs
        
        # Should set INFO level
        assert kwargs['level'] == logging.INFO
        
        # Should have correct format
        expected_format = "%(asctime)s - %(levelname)s - %(module)s - %(message)s"
        assert kwargs['format'] == expected_format
        
        # Should have handlers
        assert 'handlers' in kwargs
        handlers = kwargs['handlers']
        assert len(handlers) == 1  # Only StreamHandler (FileHandler commented out)
        assert isinstance(handlers[0], logging.StreamHandler)
    
    def test_setup_logging_returns_logger(self):
        """Test that setup_logging returns a working logger."""
        with patch('src.utils.logging_config.os.makedirs'), \
             patch('src.utils.logging_config.logging.basicConfig'):
            
            result_logger = setup_logging()
            
            # Should be a logger instance
            assert isinstance(result_logger, logging.Logger)
            
            # Should have the correct name (module name)
            assert result_logger.name == 'src.utils.logging_config'
    
    @patch('src.utils.logging_config.os.makedirs')
    def test_setup_logging_makedirs_error(self, mock_makedirs):
        """Test setup_logging handles makedirs errors gracefully."""
        mock_makedirs.side_effect = OSError("Permission denied")
        
        # Should not raise error - makedirs with exist_ok=True should handle this
        # but let's test that we handle any potential issues
        with patch('src.utils.logging_config.logging.basicConfig'):
            logger_result = setup_logging()
            assert isinstance(logger_result, logging.Logger)


class TestLoggingIntegration:
    """Integration tests for logging functionality."""
    
    def test_logger_module_variable(self):
        """Test that the module logger variable is properly initialized."""
        # The logger should be created when module is imported
        assert logger is not None
        assert isinstance(logger, logging.Logger)
    
    def test_logger_can_log_messages(self):
        """Test that the logger can actually log messages."""
        with patch('logging.Logger._log') as mock_log:
            logger.info("Test message")
            
            # Should have called the underlying log method
            mock_log.assert_called_once()
            
            # Check the call arguments
            args = mock_log.call_args[0]
            assert args[0] == logging.INFO  # Level
            assert args[1] == "Test message"  # Message
    
    def test_logger_different_levels(self):
        """Test logging at different levels."""
        with patch('logging.Logger._log') as mock_log:
            # Test different log levels
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            # Should have made 5 calls
            assert mock_log.call_count == 5
            
            # Check that correct levels were used
            calls = mock_log.call_args_list
            levels = [call[0][0] for call in calls]
            expected_levels = [
                logging.DEBUG, logging.INFO, logging.WARNING,
                logging.ERROR, logging.CRITICAL
            ]
            assert levels == expected_levels
    
    def test_logger_with_exception_info(self):
        """Test logging with exception information."""
        with patch('logging.Logger._log') as mock_log:
            try:
                raise ValueError("Test exception")
            except ValueError:
                logger.exception("An error occurred")
            
            # Should have called log
            mock_log.assert_called_once()
            
            # Should have logged at ERROR level
            args = mock_log.call_args[0]
            assert args[0] == logging.ERROR
            assert args[1] == "An error occurred"
            
            # Should have exc_info in kwargs
            kwargs = mock_log.call_args[1]
            assert kwargs.get('exc_info') is True


class TestLoggingConfiguration:
    """Test logging configuration details."""
    
    def test_logging_format_string(self):
        """Test that the logging format string is correct."""
        with patch('src.utils.logging_config.logging.basicConfig') as mock_config:
            setup_logging()
            
            # Extract format from call
            kwargs = mock_config.call_args[1]
            format_string = kwargs['format']
            
            # Should include all expected components
            expected_components = ['%(asctime)s', '%(levelname)s', '%(module)s', '%(message)s']
            for component in expected_components:
                assert component in format_string
    
    def test_handler_configuration(self):
        """Test that handlers are configured correctly."""
        with patch('src.utils.logging_config.os.makedirs'), \
             patch('src.utils.logging_config.logging.basicConfig') as mock_config:
            
            setup_logging()
            
            # Extract handlers from call
            kwargs = mock_config.call_args[1]
            handlers = kwargs['handlers']
            
            # Should have StreamHandler for console output
            stream_handlers = [h for h in handlers if isinstance(h, logging.StreamHandler)]
            assert len(stream_handlers) >= 1
            
            # FileHandler is commented out in the current implementation
            file_handlers = [h for h in handlers if isinstance(h, logging.FileHandler)]
            assert len(file_handlers) == 0  # Currently no file handler
    
    def test_log_level_configuration(self):
        """Test that log level is set correctly."""
        with patch('src.utils.logging_config.os.makedirs'), \
             patch('src.utils.logging_config.logging.basicConfig') as mock_config:
            
            setup_logging()
            
            # Extract level from call
            kwargs = mock_config.call_args[1]
            level = kwargs['level']
            
            # Should be INFO level
            assert level == logging.INFO


class TestLoggingDirectoryHandling:
    """Test logging directory creation and handling."""
    
    @patch('src.utils.logging_config.LOGS_DIR')
    @patch('src.utils.logging_config.os.makedirs')
    def test_logs_directory_creation(self, mock_makedirs, mock_logs_dir):
        """Test that logs directory is created correctly."""
        mock_logs_dir.return_value = "custom_logs"
        
        with patch('src.utils.logging_config.logging.basicConfig'):
            setup_logging()
        
        # Should call makedirs with correct path and exist_ok=True
        mock_makedirs.assert_called_once_with("custom_logs", exist_ok=True)
    
    def test_logs_directory_exists_ok(self):
        """Test that existing logs directory doesn't cause issues."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logs_dir = os.path.join(temp_dir, "logs")
            os.makedirs(logs_dir)  # Create directory first
            
            with patch('src.utils.logging_config.LOGS_DIR', logs_dir), \
                 patch('src.utils.logging_config.logging.basicConfig'):
                
                # Should not raise error even if directory exists
                logger_result = setup_logging()
                assert isinstance(logger_result, logging.Logger)


class TestRealLoggingBehavior:
    """Test actual logging behavior without mocking."""
    
    def test_actual_logger_creation(self):
        """Test that logger is actually created and usable."""
        # Import should create the logger
        from src.utils.logging_config import logger as test_logger
        
        assert test_logger is not None
        assert isinstance(test_logger, logging.Logger)
        assert hasattr(test_logger, 'info')
        assert hasattr(test_logger, 'error')
        assert hasattr(test_logger, 'warning')
        assert hasattr(test_logger, 'debug')
        assert hasattr(test_logger, 'critical')
    
    def test_logger_is_singleton(self):
        """Test that multiple imports return the same logger."""
        from src.utils.logging_config import logger as logger1
        from src.utils.logging_config import logger as logger2
        
        # Should be the same object
        assert logger1 is logger2
    
    def test_setup_logging_multiple_calls(self):
        """Test that multiple calls to setup_logging don't break anything."""
        with patch('src.utils.logging_config.os.makedirs'), \
             patch('src.utils.logging_config.logging.basicConfig') as mock_config:
            
            logger1 = setup_logging()
            logger2 = setup_logging()
            
            # Should call basicConfig multiple times
            assert mock_config.call_count == 2
            
            # Loggers should still be valid
            assert isinstance(logger1, logging.Logger)
            assert isinstance(logger2, logging.Logger)


class TestLoggingErrorHandling:
    """Test error handling in logging setup."""
    
    @patch('src.utils.logging_config.logging.basicConfig')
    @patch('src.utils.logging_config.os.makedirs')
    def test_makedirs_permission_error(self, mock_makedirs, mock_basic_config):
        """Test handling of permission errors during directory creation."""
        mock_makedirs.side_effect = PermissionError("Access denied")
        
        # Should still complete setup (exist_ok=True should handle this)
        # If it doesn't handle it, we want to know
        try:
            result_logger = setup_logging()
            # If we get here, the error was handled gracefully
            assert isinstance(result_logger, logging.Logger)
        except PermissionError:
            # If the error propagates, that's also valid behavior to test
            pytest.fail("PermissionError not handled in setup_logging")
    
    @patch('src.utils.logging_config.os.makedirs')
    @patch('src.utils.logging_config.logging.basicConfig')
    def test_logging_config_error(self, mock_basic_config, mock_makedirs):
        """Test handling of logging configuration errors."""
        mock_basic_config.side_effect = Exception("Logging config failed")
        
        # setup_logging might propagate this error, which is reasonable
        with pytest.raises(Exception, match="Logging config failed"):
            setup_logging()


if __name__ == "__main__":
    pytest.main([__file__])