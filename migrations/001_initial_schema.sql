SET XACT_ABORT ON;

IF DB_NAME() <> N'MorningTalk'
    THROW 51000, 'Run this migration only in the MorningTalk database.', 1;

IF OBJECT_ID(N'dbo.AudioFile', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.AudioFile (
        AudioFileId bigint IDENTITY(1,1) NOT NULL CONSTRAINT PK_AudioFile PRIMARY KEY,
        SourcePath nvarchar(1024) NOT NULL,
        SourcePathHash binary(32) NOT NULL,
        SourceFingerprint char(64) NOT NULL,
        SourceSizeBytes bigint NOT NULL,
        SourceModifiedUtc datetime2(3) NOT NULL,
        Status varchar(24) NOT NULL,
        AttemptCount int NOT NULL CONSTRAINT DF_AudioFile_AttemptCount DEFAULT (0),
        LeaseToken uniqueidentifier NULL,
        LeaseExpiresUtc datetime2(3) NULL,
        NextRetryUtc datetime2(3) NULL,
        PdfPath nvarchar(1024) NULL,
        LastErrorClass varchar(128) NULL,
        CreatedUtc datetime2(3) NOT NULL CONSTRAINT DF_AudioFile_CreatedUtc DEFAULT (SYSUTCDATETIME()),
        UpdatedUtc datetime2(3) NOT NULL CONSTRAINT DF_AudioFile_UpdatedUtc DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT UQ_AudioFile_SourcePathHash UNIQUE (SourcePathHash),
        CONSTRAINT CK_AudioFile_Status CHECK (Status IN
            ('discovered', 'modified', 'processing', 'completed', 'failed', 'review_required')),
        CONSTRAINT CK_AudioFile_Size CHECK (SourceSizeBytes >= 0),
        CONSTRAINT CK_AudioFile_Attempts CHECK (AttemptCount >= 0)
    );
END;

IF OBJECT_ID(N'dbo.TranscriptionChunk', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.TranscriptionChunk (
        TranscriptionChunkId bigint IDENTITY(1,1) NOT NULL CONSTRAINT PK_TranscriptionChunk PRIMARY KEY,
        AudioFileId bigint NOT NULL,
        ChunkIndex int NOT NULL,
        StartSeconds decimal(12,3) NOT NULL,
        EndSeconds decimal(12,3) NOT NULL,
        Status varchar(16) NOT NULL,
        Transcript nvarchar(max) NULL,
        CharacterCount int NOT NULL CONSTRAINT DF_TranscriptionChunk_CharacterCount DEFAULT (0),
        ErrorClass varchar(128) NULL,
        StartedUtc datetime2(3) NULL,
        CompletedUtc datetime2(3) NULL,
        CONSTRAINT FK_TranscriptionChunk_AudioFile FOREIGN KEY (AudioFileId)
            REFERENCES dbo.AudioFile(AudioFileId) ON DELETE CASCADE,
        CONSTRAINT UQ_TranscriptionChunk_FileIndex UNIQUE (AudioFileId, ChunkIndex),
        CONSTRAINT CK_TranscriptionChunk_Range CHECK
            (ChunkIndex >= 0 AND StartSeconds >= 0 AND EndSeconds > StartSeconds),
        CONSTRAINT CK_TranscriptionChunk_Status CHECK (Status IN ('pending', 'processing', 'completed', 'failed'))
    );
END;

IF OBJECT_ID(N'dbo.ProcessingLog', N'U') IS NULL
BEGIN
    CREATE TABLE dbo.ProcessingLog (
        ProcessingLogId bigint IDENTITY(1,1) NOT NULL CONSTRAINT PK_ProcessingLog PRIMARY KEY,
        AudioFileId bigint NULL,
        EventType varchar(48) NOT NULL,
        Status varchar(24) NOT NULL,
        ErrorClass varchar(128) NULL,
        Details nvarchar(1000) NULL,
        CreatedUtc datetime2(3) NOT NULL CONSTRAINT DF_ProcessingLog_CreatedUtc DEFAULT (SYSUTCDATETIME()),
        CONSTRAINT FK_ProcessingLog_AudioFile FOREIGN KEY (AudioFileId)
            REFERENCES dbo.AudioFile(AudioFileId) ON DELETE SET NULL
    );
END;

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.AudioFile') AND name = N'IX_AudioFile_StatusRetry'
)
    CREATE INDEX IX_AudioFile_StatusRetry ON dbo.AudioFile(Status, NextRetryUtc, LeaseExpiresUtc);

IF NOT EXISTS (
    SELECT 1 FROM sys.indexes
    WHERE object_id = OBJECT_ID(N'dbo.TranscriptionChunk') AND name = N'IX_TranscriptionChunk_Status'
)
    CREATE INDEX IX_TranscriptionChunk_Status ON dbo.TranscriptionChunk(AudioFileId, Status);
