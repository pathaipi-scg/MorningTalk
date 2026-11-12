SET XACT_ABORT ON;

IF DB_NAME() <> N'MorningTalk'
    THROW 51000, 'Run this migration only in the MorningTalk database.', 1;

IF TRY_CONVERT(int, SERVERPROPERTY('ProductMajorVersion')) <> 17
    THROW 51001, 'Run this migration only on SQL Server 2025 (major version 17).', 1;

BEGIN TRY
    BEGIN TRANSACTION;

    IF COL_LENGTH(N'dbo.AudioFile', N'MeetingAnalysisJson') IS NULL
        ALTER TABLE dbo.AudioFile ADD MeetingAnalysisJson nvarchar(max) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'MeetingAnalysisStatus') IS NULL
        ALTER TABLE dbo.AudioFile ADD MeetingAnalysisStatus varchar(24) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'SummaryPdfPath') IS NULL
        ALTER TABLE dbo.AudioFile ADD SummaryPdfPath nvarchar(1024) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'SummaryPdfStatus') IS NULL
        ALTER TABLE dbo.AudioFile ADD SummaryPdfStatus varchar(24) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'FullTranscriptPdfPath') IS NULL
        ALTER TABLE dbo.AudioFile ADD FullTranscriptPdfPath nvarchar(1024) NULL;

    IF COL_LENGTH(N'dbo.AudioFile', N'FullTranscriptPdfStatus') IS NULL
        ALTER TABLE dbo.AudioFile ADD FullTranscriptPdfStatus varchar(24) NULL;

    COMMIT TRANSACTION;
END TRY
BEGIN CATCH
    IF XACT_STATE() <> 0
        ROLLBACK TRANSACTION;
    THROW;
END CATCH;
