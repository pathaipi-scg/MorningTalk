SET XACT_ABORT ON;

IF DB_NAME() <> N'MorningTalk'
    THROW 51000, 'Run this migration only in the MorningTalk database.', 1;

IF TRY_CONVERT(int, SERVERPROPERTY('ProductMajorVersion')) <> 17
    THROW 51001, 'Run this migration only on SQL Server 2025 (major version 17).', 1;

BEGIN TRY
    BEGIN TRANSACTION;

    IF COL_LENGTH(N'dbo.AudioFile', N'RawTranscript') IS NULL
        ALTER TABLE dbo.AudioFile ADD RawTranscript nvarchar(max) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'CleanedTranscript') IS NULL
        ALTER TABLE dbo.AudioFile ADD CleanedTranscript nvarchar(max) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'TranscriptSegmentsJson') IS NULL
        ALTER TABLE dbo.AudioFile ADD TranscriptSegmentsJson nvarchar(max) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'DiarizationStatus') IS NULL
        ALTER TABLE dbo.AudioFile ADD DiarizationStatus varchar(24) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'CleanupStatus') IS NULL
        ALTER TABLE dbo.AudioFile ADD CleanupStatus varchar(24) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'ReviewReason') IS NULL
        ALTER TABLE dbo.AudioFile ADD ReviewReason varchar(128) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'InputTokens') IS NULL
        ALTER TABLE dbo.AudioFile ADD InputTokens int NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'OutputTokens') IS NULL
        ALTER TABLE dbo.AudioFile ADD OutputTokens int NULL;

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF XACT_STATE() <> 0
        ROLLBACK TRANSACTION;
    THROW;
END CATCH;
